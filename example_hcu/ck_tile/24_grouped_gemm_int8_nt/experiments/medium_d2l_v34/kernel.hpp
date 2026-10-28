// SPDX-License-Identifier: MIT
#pragma once
#include "pipeline.hpp"
#include "epilogue.hpp"
namespace ck_tile {
template<class P>
struct GroupedInt8NtMediumD2LKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto a=args[blockIdx.z].group_karg;
        if(a.M<=0 || a.N<=0 || a.K<=0)return;
        const index_t nt=(a.N+P::BN-1)/P::BN;
        const index_t cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM;
        const index_t nb=(inner/P::MGroup)*P::BN;
        if(mb>=a.M || nb>=a.N)return;
        using Policy=GroupedInt8NtMediumD2LPolicy<P>;
        __shared__ __align__(16) int8_t smem[Policy::SmemBytes];
        typename P::Mmac::CVecType c[P::MR][P::NR]{};
        GroupedInt8NtMediumD2LPipeline<P>{}(a,mb,nb,smem,c);
        GroupedInt8NtMediumD2LEpilogue<P>{}(a,mb,nb,smem,c);
    }
};
}
