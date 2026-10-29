// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_pipeline.hpp"
namespace ck_tile {
// All12 waves participate in packed loads and CTA barriers. The first8 waves
// compute a conventional4x2 MMAC grid. This supports the native path's
// unsupported strides and tails without depending on experimental providers.
struct GroupedInt8NtWaspFallbackProblem
    : GroupedInt8NtTileProblem<256,256,64,4,2,false,false>
{
    static constexpr index_t Threads=768,ComputeThreads=512;
};
template<class P> struct GroupedInt8NtWaspFallback
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem) const
    {
        using F=GroupedInt8NtWaspFallbackProblem;
        using Base=GroupedInt8NtTilePipeline<F>;
        static_assert(GroupedInt8NtTilePolicy<F>::SmemBytes<=65536);
        typename F::Mmac::CVecType c[F::MR][F::NR]{};
        Base pipeline;
        auto current=pipeline.Load(a,mb,nb,0);
        pipeline.Stage(current,smem);
        block_sync_lds();
        for(index_t kb=0;kb<a.K;kb+=F::BK)
        {
            typename Base::GlobalTile next{};
            if(kb+F::BK<a.K)next=pipeline.Load(a,mb,nb,kb+F::BK);
            if(get_thread_id()<F::ComputeThreads)pipeline.Compute(smem,c);
            block_sync_lds();
            if(kb+F::BK<a.K)
            {
                pipeline.Stage(next,smem);
                block_sync_lds();
            }
        }
        const index_t tid=get_thread_id();
        if(tid>=F::ComputeThreads)return;
        const index_t lane=tid%64,wave=tid/64,wm=wave/F::WN,wn=wave%F::WN;
        static_for<0,F::MR,1>{}([&](auto mm){constexpr index_t mi=decltype(mm)::value;
            const index_t row=mb+wm*(F::BM/F::WM)+mi*16+lane%16;
            static_for<0,F::NR,1>{}([&](auto nn){constexpr index_t ni=decltype(nn)::value;
                static_for<0,4,1>{}([&](auto vv){constexpr index_t vi=decltype(vv)::value;
                    const index_t col=nb+wn*(F::BN/F::WN)+ni*16+lane/16+vi*4;
                    if(row<a.M && col<a.N)
                    {
                        const long long offset=P::Transpose?static_cast<long long>(col)*a.stride_E+row:static_cast<long long>(row)*a.stride_E+col;
                        static_cast<int32_t*>(a.e_ptr)[offset]=c[mi][ni][vi];
                    }
                });
            });
        });
    }
};
}
