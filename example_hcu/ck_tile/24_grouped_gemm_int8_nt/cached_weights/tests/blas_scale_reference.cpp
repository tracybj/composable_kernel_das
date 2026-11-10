// SPDX-License-Identifier: MIT
// Test-only native BLASLt reference. This is not an implementation fallback.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <dlfcn.h>
struct ScaleReferenceGroup
{
    const void *a, *b;
    void* c;
    const float *scale_a, *scale_b;
    int m, n, k, sa, sb, sc;
};
static thread_local std::string message;
static void check(hipblasStatus_t s)
{
    if(s)
        throw std::runtime_error("BLAS status " + std::to_string(s));
}
static void hip_check(hipError_t s)
{
    if(s)
        throw std::runtime_error(hipGetErrorString(s));
}
extern "C" const char* blas_scale_reference_error() { return message.c_str(); }
extern "C" const char* blas_scale_reference_loaded_path()
{
    static Dl_info info{};
    dladdr(reinterpret_cast<void*>(&hipblasLtCreate), &info);
    return info.dli_fname;
}
extern "C" int blas_scale_reference(int g, const ScaleReferenceGroup* ds, int output)
{
    hipblasLtHandle_t handle = nullptr;
    void *workspace = nullptr, *device_args = nullptr;
    try
    {
        check(hipblasLtCreate(&handle));
        const auto dtype = output == 1 ? HIP_R_16BF : HIP_R_16F;
        auto make        = [&] {
            return std::make_unique<hipblaslt_ext::GroupedGemm>(handle,
                                                                HIPBLAS_OP_T,
                                                                HIPBLAS_OP_N,
                                                                HIP_R_8I,
                                                                HIP_R_8I,
                                                                dtype,
                                                                dtype,
                                                                HIPBLAS_COMPUTE_32F);
        };
        auto gg     = make();
        float alpha = 1, beta = 0;
        std::vector<int64_t> m(g), n(g), k(g), batch(g, 1), lda(g), ldb(g), ld(g), sa(g), sb(g),
            sc(g);
        std::vector<hipblaslt_ext::GemmEpilogue> ep(g);
        std::vector<hipblaslt_ext::GemmInputs> inputs(g);
        for(int i = 0; i < g; i++)
        {
            const auto& d = ds[i];
            m[i]          = d.n;
            n[i]          = d.m;
            k[i]          = d.k;
            lda[i]        = d.sb;
            ldb[i]        = d.sa;
            ld[i]         = d.sc;
            sa[i]         = int64_t(d.n) * d.sb;
            sb[i]         = int64_t(d.m) * d.sa;
            sc[i]         = int64_t(d.m) * d.sc;
            ep[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
            inputs[i].setA(d.b);
            inputs[i].setB(d.a);
            inputs[i].setC(d.c);
            inputs[i].setD(d.c);
            inputs[i].setAlpha(&alpha);
            inputs[i].setBeta(&beta);
            inputs[i].setScaleA(d.scale_b);
            inputs[i].setScaleB(d.scale_a);
        }
        hipblaslt_ext::GemmProblemType type(
            HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_8I, HIP_R_8I, dtype, dtype, HIPBLAS_COMPUTE_32F);
        auto set = [&] {
            for(int i = 0; i < g; i++)
            {
                m[i] = ds[i].n;
                n[i] = ds[i].m;
                k[i] = ds[i].k;
            }
            check(
                gg->setProblem(m, n, k, batch, lda, ldb, ld, ld, sa, sb, sc, sc, ep, inputs, type));
        };
        set();
        std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
        check(hipblaslt_ext::getAllAlgos(handle,
                                         hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                         HIPBLAS_OP_T,
                                         HIPBLAS_OP_N,
                                         HIP_R_8I,
                                         HIP_R_8I,
                                         dtype,
                                         dtype,
                                         HIPBLAS_COMPUTE_32F,
                                         candidates));
        bool found = false;
        for(auto candidate : candidates)
        {
            size_t ws                          = 0;
            candidate.algo.max_workspace_bytes = 256ULL << 20;
            if(gg->isAlgoSupported(candidate.algo, ws) != HIPBLAS_STATUS_SUCCESS ||
               ws > (256ULL << 20))
                continue;
            gg = make();
            set();
            std::vector<hipblaslt_ext::UserArguments> args(g);
            check(gg->getDefaultValueForDeviceUserArguments(args.data()));
            for(int i = 0; i < g; i++)
                if(args[i].m != ds[i].n || args[i].n != ds[i].m ||
                   args[i].scaleA != ds[i].scale_b || args[i].scaleB != ds[i].scale_a)
                    throw std::runtime_error("BLAS scale mapping changed");
            hip_check(hipMalloc(&device_args, args.size() * sizeof(args[0])));
            hip_check(hipMemcpy(
                device_args, args.data(), args.size() * sizeof(args[0]), hipMemcpyHostToDevice));
            if(ws)
                hip_check(hipMalloc(&workspace, ws));
            check(gg->initialize(candidate.algo, workspace, true));
            check(gg->run(device_args, nullptr));
            hip_check(hipDeviceSynchronize());
            found = true;
            break;
        }
        if(!found)
            throw std::runtime_error("NO_SUPPORTED_NATIVE_SCALED_ALGO");
        gg.reset();
        if(workspace)
            hip_check(hipFree(workspace));
        if(device_args)
            hip_check(hipFree(device_args));
        check(hipblasLtDestroy(handle));
        return 0;
    }
    catch(const std::exception& e)
    {
        message = e.what();
        if(workspace)
            (void)hipFree(workspace);
        if(device_args)
            (void)hipFree(device_args);
        if(handle)
            (void)hipblasLtDestroy(handle);
        return -1;
    }
}
