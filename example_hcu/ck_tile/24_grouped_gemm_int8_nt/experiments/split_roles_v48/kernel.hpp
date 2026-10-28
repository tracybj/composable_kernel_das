// SPDX-License-Identifier: MIT
#pragma once
#include "pipeline.hpp"
#include "../dense_k128_v46/kernel.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtSplitRolesKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto a=args[blockIdx.z].group_karg;if(a.M<=0 || a.N<=0 || a.K<=0)return;
        const index_t nt=(a.N+P::BN-1)/P::BN,cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM,nb=(inner/P::MGroup)*P::BN;
        if(mb>=a.M || nb>=a.N)return;
        __shared__ __align__(16) int8_t smem[65536];
        // Include masked K padding in the i32-offset proof, even for invalid loads.
        const long long padded_k=(static_cast<long long>(a.K)+127)/128*128;
        const bool strides=static_cast<long long>(255)*a.stride_As[0]+padded_k<=0x7fffffffLL && static_cast<long long>(255)*a.stride_Bs[0]+padded_k<=0x7fffffffLL;
        const bool fast=strides && (!P::Aligned || (a.M%256==0 && a.N%256==0 && a.K%128==0));
        if(!fast)
        {
            using F=GroupedInt8NtDenseKFallback<P>;
            typename P::Mmac::CVecType accum[P::MR][P::NR]{};
            GroupedInt8NtD2LGeometryPipeline<F>{}(a,mb,nb,smem,accum);
            GroupedInt8NtNoLdsEpilogue<F,true>{}(a,mb,nb,accum);
            return;
        }
        const index_t wave=__builtin_amdgcn_readfirstlane(get_thread_id()/64);
        if(wave>=P::ComputeThreads/64)GroupedInt8NtProducerLoop<P>{}(a,mb,nb,smem);
        else GroupedInt8NtConsumerLoop<P>{}(a,mb,nb,smem);
    }
};
}
