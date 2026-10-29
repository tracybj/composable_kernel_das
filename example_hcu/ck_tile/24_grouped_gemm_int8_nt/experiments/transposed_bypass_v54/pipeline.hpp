// SPDX-License-Identifier: MIT
#pragma once
#include "epilogue.hpp"
#include "../bypass_b_ordered_v52/pipeline.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtTransposedConsumer
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem) const
    {
        const index_t tid=get_thread_id();__builtin_assume(tid<P::ComputeThreads);
        GroupedInt8NtBypassBOrderedPipeline<P> pipe;
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const long long bytes=static_cast<long long>(a.N-nb-1)*a.stride_Bs[0]+a.K;
        const auto br=make_wave_buffer_resource(bp,static_cast<uint32_t>(bytes<0x7fffffffLL?bytes:0x7fffffffLL));
        typename P::Mmac::CVecType accum[P::MR][P::NR]{};
        auto current=pipe.LoadB(a,nb,0,br);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
        index_t slot=0;
        for(index_t kb=0;kb+128<a.K;kb+=128)
        {
            if constexpr(P::PrefetchB)
            {
                auto next=pipe.LoadB(a,nb,kb+128,br);
                if constexpr(P::ConsumerPriority)asm volatile("s_setprio 1" ::: "memory");
                pipe.Compute(smem+slot*32768,current,accum);
                if constexpr(P::ConsumerPriority)asm volatile("s_setprio 0" ::: "memory");
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                block_sync_lds();current=next;
            }
            else
            {
                pipe.Compute(smem+slot*32768,current,accum);block_sync_lds();
                current=pipe.LoadB(a,nb,kb+128,br);asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            }
            slot^=1;
        }
        pipe.Compute(smem+slot*32768,current,accum);
        GroupedInt8NtTransposedEpilogue<P>{}(a,mb,nb,accum);
    }
};
}
