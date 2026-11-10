// SPDX-License-Identifier: MIT
// Benchmark setup owns allocations; timed launch calls only the public v1 ABI.
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
struct Desc
{
    const void* a;
    const void* b;
    void* c;
    int kb, m, n, k, sa, sb, sc, nd;
    const void* ds;
    const void* sds;
};
namespace {
std::string error;
void hs(hipError_t s)
{
    if(s != hipSuccess)
        throw std::runtime_error(hipGetErrorString(s));
}
struct Context
{
    void* args = nullptr;
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    ck_tile_hcu_grouped_gemm_launch_v1 launch{};
    hipEvent_t start{}, stop{};
    ~Context()
    {
        if(args)
            (void)hipFree(args);
        if(start)
            (void)hipEventDestroy(start);
        if(stop)
            (void)hipEventDestroy(stop);
    }
    void run()
    {
        const int s = ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &launch, nullptr);
        if(s)
            throw std::runtime_error("public run failed: " + std::to_string(s));
    }
};
} // namespace
extern "C" const char* int8_error() { return error.c_str(); }
extern "C" const char* int8_name(void* c)
{
    return static_cast<Context*>(c)->selection.instance_name;
}
extern "C" unsigned int8_instance(void* c)
{
    return static_cast<Context*>(c)->selection.instance_id;
}
extern "C" void* int8_create(const Desc* ds, int g)
{
    try
    {
        if(!ds || g < 1)
            throw std::runtime_error("invalid group count");
        auto p            = std::make_unique<Context>();
        auto& pr          = p->problem;
        pr.struct_size    = sizeof(pr);
        pr.abi_version    = 1;
        pr.data_type      = CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_INT8_INT32_V1;
        pr.layout         = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
        pr.group_count    = g;
        long long total_m = 0;
        bool same = true, empty = false;
        std::vector<ck_tile::GemmTransKernelArg> args;
        for(int i = 0; i < g; ++i)
        {
            const auto& d = ds[i];
            if(d.m < 0 || d.k != ds[0].k || d.n != ds[0].n || d.kb != 1 || d.nd != 0 ||
               d.sa < d.k || d.sb < d.k || d.sc < d.n || d.sa % 16 || d.sb % 16 || d.sc % 4 ||
               (reinterpret_cast<uintptr_t>(d.a) % 16) || (reinterpret_cast<uintptr_t>(d.b) % 16) ||
               (reinterpret_cast<uintptr_t>(d.c) % 16))
                throw std::runtime_error("unsupported descriptor/alignment");
            same &= d.m == ds[0].m;
            empty |= d.m == 0;
            total_m += d.m;
            ck_tile::UniversalGemmKernelArgs<1, 1, 0> a{
                {d.a}, {d.b}, {}, d.c, d.m, d.n, d.k, {d.sa}, {d.sb}, {}, d.sc, 1};
            args.emplace_back(std::move(a));
        }
        pr.m = {same ? 0u : 1u, 0, same ? std::max(ds[0].m, 1) : std::max(total_m, 1LL)};
        pr.n = {0, 0, ds[0].n};
        pr.k = {0, 0, ds[0].k};
        pr.problem_flags         = (empty ? 1u : 0u) | (same ? 0u : 2u);
        p->selection.struct_size = sizeof(p->selection);
        p->selection.abi_version = 1;
        int s = ck_tile_hcu_grouped_gemm_select_device_args_v1(&pr, &p->selection);
        if(s)
            throw std::runtime_error("public select failed: " + std::to_string(s));
        ck_tile_hcu_grouped_gemm_candidate_info_v1 info{};
        info.struct_size = sizeof(info);
        info.abi_version = 1;
        s = ck_tile_hcu_grouped_gemm_query_instance_v1(p->selection.instance_id, &info);
        if(s || info.data_type_mask != CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_MASK_INT8_INT32_V1 ||
           info.layout_mask != CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1)
            throw std::runtime_error("public candidate contract mismatch");
        hs(hipMalloc(&p->args, args.size() * sizeof(args[0])));
        hs(hipMemcpy(p->args, args.data(), args.size() * sizeof(args[0]), hipMemcpyHostToDevice));
        p->launch.struct_size          = sizeof(p->launch);
        p->launch.abi_version          = 1;
        p->launch.device_args          = p->args;
        p->launch.expected_instance_id = p->selection.instance_id;
        hs(hipEventCreate(&p->start));
        hs(hipEventCreate(&p->stop));
        return p.release();
    }
    catch(const std::exception& e)
    {
        error = e.what();
        return nullptr;
    }
}
extern "C" float int8_run(void* c, int warm, int repeat)
{
    try
    {
        if(!c || warm < 0 || repeat < 1)
            return -1;
        auto& p = *static_cast<Context*>(c);
        for(int i = 0; i < warm; ++i)
            p.run();
        auto stream = reinterpret_cast<hipStream_t>(p.launch.stream);
        hs(hipEventRecord(p.start, stream));
        for(int i = 0; i < repeat; ++i)
            p.run();
        hs(hipEventRecord(p.stop, stream));
        hs(hipEventSynchronize(p.stop));
        float ms;
        hs(hipEventElapsedTime(&ms, p.start, p.stop));
        return ms / repeat;
    }
    catch(const std::exception& e)
    {
        error = e.what();
        return -100;
    }
}
extern "C" void int8_destroy(void* c) { delete static_cast<Context*>(c); }
extern "C" int int8_mismatch_check(void* c)
{
    auto& p = *static_cast<Context*>(c);
    auto l  = p.launch;
    l.expected_instance_id += 100;
    return ck_tile_hcu_grouped_gemm_run_device_args_v1(&p.problem, &l, nullptr);
}
extern "C" void int8_set_stream(void* c, void* stream)
{
    static_cast<Context*>(c)->launch.stream = stream;
}
extern "C" int int8_contract_check(void* c, int kind)
{
    auto& p = *static_cast<Context*>(c);
    auto pr = p.problem;
    auto l  = p.launch;
    if(kind == 0)
        l.num_cu = 1;
    if(kind == 1)
        pr.layout = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1;
    if(kind == 2)
        pr.abi_version = 2;
    if(kind == 3)
        pr.struct_size = 4;
    if(kind == 4)
        pr.reserved[0] = 1;
    return ck_tile_hcu_grouped_gemm_run_device_args_v1(&pr, &l, nullptr);
}
