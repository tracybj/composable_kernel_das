// Native hipBLASLt GroupedGemm: INT8 A/B, INT32 accumulation/output, null scales.
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
struct Desc
{
    const void* a;
    const void* b;
    void* c;
    int kb, m, n, k, sa, sb, sc, nd;
    const void* ds;
    const void* sds;
};
static std::string error;
static void hs(hipError_t s, const char* w)
{
    if(s != hipSuccess)
        throw std::runtime_error(std::string(w) + ": " + hipGetErrorString(s));
}
static void bs(hipblasStatus_t s, const char* w)
{
    if(s != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(w) + ": " + std::to_string(s));
}
struct Context
{
    int32_t alpha = 1, beta = 0;
    hipblasLtHandle_t handle{};
    std::unique_ptr<hipblaslt_ext::GroupedGemm> gg;
    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
    std::vector<std::string> origins;
    void* workspace{};
    void* user_args{};
    size_t bytes = 0;
    int selected = -1;
    hipEvent_t start{}, stop{};
    ~Context()
    {
        (void)hipDeviceSynchronize();
        gg.reset();
        if(handle)
            (void)hipblasLtDestroy(handle);
        if(workspace)
            (void)hipFree(workspace);
        if(user_args)
            (void)hipFree(user_args);
        if(start)
            (void)hipEventDestroy(start);
        if(stop)
            (void)hipEventDestroy(stop);
    }
    void launch() { bs(gg->run(user_args, nullptr), "run device user args"); }
};
extern "C" const char* blas_error() { return error.c_str(); }
extern "C" const char* blas_loaded_path()
{
    static Dl_info d{};
    dladdr(reinterpret_cast<void*>(&hipblasLtCreate), &d);
    return d.dli_fname;
}
extern "C" void* blas_create(int g, const Desc* ds)
{
    try
    {
        if(g < 1 || !ds)
            throw std::runtime_error("empty group");
        auto p = std::make_unique<Context>();
        bs(hipblasLtCreate(&p->handle), "create handle");
        auto make = [&]() {
            return std::make_unique<hipblaslt_ext::GroupedGemm>(p->handle,
                                                                HIPBLAS_OP_T,
                                                                HIPBLAS_OP_N,
                                                                HIP_R_8I,
                                                                HIP_R_8I,
                                                                HIP_R_32I,
                                                                HIP_R_32I,
                                                                HIPBLAS_COMPUTE_32I);
        };
        p->gg = make();
        // Row-major C=A*B^T is column-major C^T=B*A^T, without physical copies.
        std::vector<int64_t> m(g), n(g), k(g), batch(g, 1), lda(g), ldb(g), ldc(g), sa(g), sb(g),
            sc(g);
        std::vector<hipblaslt_ext::GemmEpilogue> eps(g);
        std::vector<hipblaslt_ext::GemmInputs> in(g);
        for(int i = 0; i < g; ++i)
        {
            if(ds[i].m < 1 || ds[i].n < 1 || ds[i].k < 1 || ds[i].kb != 1)
                throw std::runtime_error("invalid descriptor");
            m[i]   = ds[i].n;
            n[i]   = ds[i].m;
            k[i]   = ds[i].k;
            lda[i] = ds[i].sb;
            ldb[i] = ds[i].sa;
            ldc[i] = ds[i].sc;
            sa[i]  = int64_t(ds[i].n) * ds[i].k;
            sb[i]  = int64_t(ds[i].m) * ds[i].k;
            sc[i]  = int64_t(ds[i].m) * ds[i].n;
            eps[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
            eps[i].setBiasDataType(HIP_R_32I);
            in[i].setA(ds[i].b);
            in[i].setB(ds[i].a);
            in[i].setC(ds[i].c);
            in[i].setD(ds[i].c);
            in[i].setAlpha(&p->alpha);
            in[i].setBeta(&p->beta);
            in[i].setBias(nullptr);
            in[i].setScaleA(nullptr);
            in[i].setScaleB(nullptr);
            in[i].setScaleAlphaVec(nullptr);
        }
        hipblaslt_ext::GemmProblemType type(HIPBLAS_OP_T,
                                            HIPBLAS_OP_N,
                                            HIP_R_8I,
                                            HIP_R_8I,
                                            HIP_R_32I,
                                            HIP_R_32I,
                                            HIPBLAS_COMPUTE_32I);
        auto set_problem = [&]() {
            bs(p->gg->setProblem(m, n, k, batch, lda, ldb, ldc, ldc, sa, sb, sc, sc, eps, in, type),
               "set problem");
        };
        set_problem();
        constexpr size_t limit = 256ULL << 20;
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(limit);
        std::vector<hipblasLtMatmulHeuristicResult_t> h, all;
        (void)p->gg->algoGetHeuristic(64, pref, h);
        (void)hipblaslt_ext::getAllAlgos(p->handle,
                                         hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                         HIPBLAS_OP_T,
                                         HIPBLAS_OP_N,
                                         HIP_R_8I,
                                         HIP_R_8I,
                                         HIP_R_32I,
                                         HIP_R_32I,
                                         HIPBLAS_COMPUTE_32I,
                                         all);
        std::set<int> seen;
        auto append = [&](auto& candidates, const char* origin) {
            for(auto cand : candidates)
            {
                cand.algo.max_workspace_bytes = limit;
                int id                        = hipblaslt_ext::getIndexFromAlgo(cand.algo);
                if(id < 0 || seen.count(id))
                    continue;
                size_t ws = 0;
                if(p->gg->isAlgoSupported(cand.algo, ws) != HIPBLAS_STATUS_SUCCESS || ws > limit)
                    continue;
                seen.insert(id);
                cand.workspaceSize = ws;
                p->bytes           = std::max(p->bytes, ws);
                p->algos.push_back(cand);
                p->origins.push_back(origin);
            }
        };
        append(h, "heuristic");
        append(all, "getAllAlgos");
        if(p->algos.empty())
            throw std::runtime_error("NO_SUPPORTED_ALGO: heuristic=" + std::to_string(h.size()) +
                                     ", all=" + std::to_string(all.size()));
        // Querying can mutate execution descriptors in this library. Restore them.
        for(int i = 0; i < g; ++i)
        {
            m[i] = ds[i].n;
            n[i] = ds[i].m;
            k[i] = ds[i].k;
        }
        p->gg = make();
        set_problem();
        std::vector<hipblaslt_ext::UserArguments> args(g);
        bs(p->gg->getDefaultValueForDeviceUserArguments(args.data()), "default device args");
        for(int i = 0; i < g; ++i)
        {
            const auto& a = args[i];
            const auto& d = ds[i];
            if(a.m != d.n || a.n != d.m || a.k != d.k || a.batch != 1 || a.a != d.b || a.b != d.a ||
               a.c != d.c || a.d != d.c || a.strideA1 != d.sb || a.strideB1 != d.sa ||
               a.strideC1 != d.sc || a.strideD1 != d.sc)
                throw std::runtime_error("device arguments differ from exact group " +
                                         std::to_string(i));
        }
        hs(hipMalloc(&p->user_args, args.size() * sizeof(args[0])), "allocate args");
        hs(hipMemcpy(
               p->user_args, args.data(), args.size() * sizeof(args[0]), hipMemcpyHostToDevice),
           "copy args");
        if(p->bytes)
            hs(hipMalloc(&p->workspace, p->bytes), "workspace");
        hs(hipEventCreate(&p->start), "start event");
        hs(hipEventCreate(&p->stop), "stop event");
        return p.release();
    }
    catch(const std::exception& e)
    {
        error = e.what();
        return nullptr;
    }
}
extern "C" int blas_count(void* x) { return static_cast<Context*>(x)->algos.size(); }
extern "C" int blas_select(void* x, int i)
{
    try
    {
        auto& p = *static_cast<Context*>(x);
        if(i < 0 || i >= int(p.algos.size()))
            return -1;
        bs(p.gg->initialize(p.algos[i].algo, p.workspace, true, nullptr), "initialize");
        p.selected = i;
        return 0;
    }
    catch(const std::exception& e)
    {
        error = e.what();
        return -1;
    }
}
extern "C" const char* blas_info(void* x)
{
    static std::string s;
    auto& p = *static_cast<Context*>(x);
    if(p.selected < 0)
        return "unselected";
    s = "algo_index=" + std::to_string(hipblaslt_ext::getIndexFromAlgo(p.algos[p.selected].algo)) +
        ", source=" + p.origins[p.selected] + ", kernel=" + p.gg->getKernelName() +
        ", solution=" + p.gg->getSolutionName();
    return s.c_str();
}
extern "C" float blas_run(void* x, int warm, int repeat)
{
    try
    {
        auto& p = *static_cast<Context*>(x);
        if(warm < 0 || repeat < 1)
            return -1;
        for(int i = 0; i < warm; ++i)
            p.launch();
        hs(hipEventRecord(p.start), "record start");
        for(int i = 0; i < repeat; ++i)
            p.launch();
        hs(hipEventRecord(p.stop), "record stop");
        hs(hipEventSynchronize(p.stop), "synchronize stop");
        float ms;
        hs(hipEventElapsedTime(&ms, p.start, p.stop), "elapsed");
        return ms / repeat;
    }
    catch(const std::exception& e)
    {
        error = e.what();
        return -1;
    }
}
extern "C" void blas_destroy(void* x) { delete static_cast<Context*>(x); }
