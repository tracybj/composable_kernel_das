// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
#include "../dense_k128_v46/pipeline.hpp"
#include "../direct_epilogue_v42/epilogue.hpp"
namespace ck_tile {
// Producer and consumer use distinct loop/SSA scopes. They rendezvous at the
// same block-barrier phases: publish0, retire-current, publish-next.
template<class P> struct GroupedInt8NtProducerLoop
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem) const
    {
        const index_t tid=get_thread_id();__builtin_assume(tid>=P::ComputeThreads && tid<P::Threads);
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const long long ab=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K,bb=static_cast<long long>(a.N-nb-1)*a.stride_Bs[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(ab<0x7fffffffLL?ab:0x7fffffffLL));
        const auto br=make_wave_buffer_resource(bp,static_cast<uint32_t>(bb<0x7fffffffLL?bb:0x7fffffffLL));
        GroupedInt8NtDenseKPipeline<P> pipe;
        auto first=pipe.Load(a,mb,nb,0,ar,br);pipe.Stage(first,smem);
        block_sync_lds(); // Publish the first tile.
        for(index_t kb=0;kb+128<a.K;kb+=128)
        {
            auto next=pipe.Load(a,mb,nb,kb+128,ar,br);
            block_sync_lds(); // Consumers have retired all current LDS reads.
            pipe.Stage(next,smem);
            block_sync_lds(); // Publish the next tile.
        }
    }
};
template<class P> struct GroupedInt8NtConsumerLoop
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,const int8_t* smem) const
    {
        const index_t tid=get_thread_id();__builtin_assume(tid<P::ComputeThreads);
        using C=GroupedInt8NtDenseKProblem<0,P::Swizzle8,P::Paired,P::Aligned,P::WM,P::WN,P::MGroup,P::Fence>;
        typename P::Mmac::CVecType accum[P::MR][P::NR]{};
        GroupedInt8NtDenseKPipeline<C> pipe;
        block_sync_lds(); // First tile is readable after producers publish it.
        for(index_t kb=0;kb+128<a.K;kb+=128)
        {
            if constexpr(P::ConsumerPriority)asm volatile("s_setprio 1" ::: "memory");
            pipe.Compute(smem,accum);
            if constexpr(P::ConsumerPriority)asm volatile("s_setprio 0" ::: "memory");
            block_sync_lds(); // Retire current reads before overwrite.
            block_sync_lds(); // Wait for the next tile publication.
        }
        if constexpr(P::ConsumerPriority)asm volatile("s_setprio 1" ::: "memory");
        pipe.Compute(smem,accum);
        if constexpr(P::ConsumerPriority)asm volatile("s_setprio 0" ::: "memory");
        GroupedInt8NtNoLdsEpilogue<C,true>{}(a,mb,nb,accum);
    }
};
}
