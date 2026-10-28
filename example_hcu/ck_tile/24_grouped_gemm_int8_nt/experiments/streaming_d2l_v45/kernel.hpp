// SPDX-License-Identifier: MIT
#pragma once
#include "pipeline.hpp"
#include "epilogue.hpp"
#include "../direct_lds_v30/pipeline.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtStreamFallback : GroupedInt8NtD2LGeometryProblem<P::WM,P::WN,P::MGroup,false,1>
{
    static constexpr index_t ProducerWaves=P::ProducerWaves,ComputeThreads=P::ComputeThreads,Threads=P::Threads;
};
template<class P> struct GroupedInt8NtStreamKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto a=args[blockIdx.z].group_karg;if(a.M<=0 || a.N<=0 || a.K<=0)return;
        const index_t nt=(a.N+P::BN-1)/P::BN;
        const index_t cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM,nb=(inner/P::MGroup)*P::BN;
        if(mb>=a.M || nb>=a.N)return;
        using Policy=GroupedInt8NtStreamPolicy<P>;
        __shared__ __align__(16) int8_t smem[Policy::SmemBytes];
        typename P::Mmac::CVecType c[P::MR][P::NR]{};
        const bool fast=a.M%256==0 && a.N%256==0 && a.K%32==0 && a.K>=(P::Stages-1)*32 &&
            static_cast<long long>(255)*a.stride_As[0]+a.K<=0x7fffffffLL &&
            static_cast<long long>(255)*a.stride_Bs[0]+a.K<=0x7fffffffLL;
        if(fast)GroupedInt8NtStreamPipeline<P>{}(a,mb,nb,smem,c);
        else
        {
            // The fallback uses canonical trailing producer waves. Its epilogue
            // below uses the same canonical compute-wave map, including when
            // the fast path would use a middle producer-wave placement.
            using F=GroupedInt8NtStreamFallback<P>;
            GroupedInt8NtD2LGeometryPipeline<F>{}(a,mb,nb,smem,c);
        }
        const index_t wave=fast?P::ComputeWave(get_thread_id()):get_thread_id()/64;
        if(fast)
        {
            if(wave>=0)GroupedInt8NtStreamEpilogue<P>{}(a,mb,nb,c,wave);
        }
        else
        {
            using F=GroupedInt8NtStreamFallback<P>;
            if(wave<P::ComputeWaves)GroupedInt8NtStreamEpilogue<F>{}(a,mb,nb,c,wave);
        }
    }
};
}
