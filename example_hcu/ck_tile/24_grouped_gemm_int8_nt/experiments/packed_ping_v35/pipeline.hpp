// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_pipeline.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtPackedPingPolicy : GroupedInt8NtTilePolicy<P>
{
    using Base=GroupedInt8NtTilePolicy<P>;
    static constexpr index_t SmemBytes=2*Base::OperandBytes>Base::CStripeBytes?2*Base::OperandBytes:Base::CStripeBytes;
    static_assert(SmemBytes<=65536);
};
template<class P,class Policy=GroupedInt8NtPackedPingPolicy<P>>
struct GroupedInt8NtPackedPingPipeline : GroupedInt8NtTilePipeline<P,Policy>
{
    using Base=GroupedInt8NtTilePipeline<P,Policy>;
    using Acc=typename Base::Acc;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        auto current=this->Load(a,mb,nb,0);
        this->Stage(current,smem);block_sync_lds();
        index_t buffer=0;
        for(index_t kb=0;kb+P::BK<a.K;kb+=P::BK)
        {
            auto next=this->Load(a,mb,nb,kb+P::BK);
            __builtin_amdgcn_sched_barrier(0);
            this->Compute(smem+buffer*Policy::OperandBytes,c);
            __builtin_amdgcn_sched_barrier(0);
            this->Stage(next,smem+(1-buffer)*Policy::OperandBytes);
            // The previous iteration's barrier retired all readers of this buffer.
            // This barrier publishes the next tile and retires the current readers.
            block_sync_lds();buffer=1-buffer;
        }
        this->Compute(smem+buffer*Policy::OperandBytes,c);
    }
};
}
