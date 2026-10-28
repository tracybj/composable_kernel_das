// SPDX-License-Identifier: MIT
// Development-only candidate screening. Final acceptance uses the public ABI.
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include <vector>
#include <stdexcept>
#include <string>
struct Desc
{
    const void* a;
    const void* b;
    void* c;
    int kb, m, n, k, sa, sb, sc, nd;
    const void* ds;
    const void* sds;
};
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned, const void*, int, int, int, void*);
struct Probe
{
    void* args = nullptr;
    int g, m, n;
    hipEvent_t begin{}, end{};
    ~Probe()
    {
        if(args)
            hipFree(args);
        if(begin)
            hipEventDestroy(begin);
        if(end)
            hipEventDestroy(end);
    }
};
extern "C" void* probe_create(const Desc* ds, int g)
{
    try
    {
        auto p = new Probe{};
        p->g   = g;
        p->m   = 0;
        p->n   = 0;
        std::vector<ck_tile::GemmTransKernelArg> v;
        for(int i = 0; i < g; ++i)
        {
            const auto& d = ds[i];
            ck_tile::UniversalGemmKernelArgs<1, 1, 0> a{
                {d.a}, {d.b}, {}, d.c, d.m, d.n, d.k, {d.sa}, {d.sb}, {}, d.sc, 1};
            v.emplace_back(std::move(a));
            p->m = std::max(p->m, d.m);
            p->n = std::max(p->n, d.n);
        }
        if(hipMalloc(&p->args, v.size() * sizeof(v[0])) != hipSuccess)
        {
            delete p;
            return nullptr;
        }
        hipMemcpy(p->args, v.data(), v.size() * sizeof(v[0]), hipMemcpyHostToDevice);
        hipEventCreate(&p->begin);
        hipEventCreate(&p->end);
        return p;
    }
    catch(...)
    {
        return nullptr;
    }
}
extern "C" float probe_run(void* c, int id, int warm, int repeat)
{
    auto& p = *static_cast<Probe*>(c);
    for(int i = 0; i < warm; ++i)
        if(ck_tile_grouped_int8_nt_launch(id, p.args, p.g, p.m, p.n, nullptr))
            return -1;
    hipEventRecord(p.begin);
    for(int i = 0; i < repeat; ++i)
        if(ck_tile_grouped_int8_nt_launch(id, p.args, p.g, p.m, p.n, nullptr))
            return -2;
    hipEventRecord(p.end);
    if(hipEventSynchronize(p.end) != hipSuccess)
        return -3;
    float ms = 0;
    hipEventElapsedTime(&ms, p.begin, p.end);
    return ms / repeat;
}
extern "C" void probe_destroy(void* p) { delete static_cast<Probe*>(p); }
