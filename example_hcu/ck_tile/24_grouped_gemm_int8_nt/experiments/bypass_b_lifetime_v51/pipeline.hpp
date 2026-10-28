// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "../direct_epilogue_v42/epilogue.hpp"
namespace ck_tile {
// A uses two alternating 32KiB LDS slots. B is loaded by its consumer wave
// directly in the MMAC K64-pair permutation; it never occupies LDS.
template<class P,class Policy=GroupedInt8NtBypassBLifetimePolicy<P>>
struct GroupedInt8NtBypassBLifetimePipeline
{
    using Copy=typename P::CopyVector;using Acc=typename P::Mmac::CVecType;
    struct BTile {Copy values[2][P::NR];};
    CK_TILE_DEVICE static index_t Swizzle(index_t row){return ((row>>P::AShift)&7)*16;}
    CK_TILE_DEVICE void StageA(const UniversalGemmKernelArgs<>& a,index_t mb,index_t kb,
                              int8_t* dst,const int32x4_t& ar) const
    {
        const index_t stid=get_thread_id()-P::ComputeThreads;
        __builtin_assume(stid>=0 && stid<Policy::StageThreads);
        static_for<0,Policy::AIssues,1>{}([&](auto issue){
            constexpr index_t j=decltype(issue)::value;
            const index_t vec=stid+j*Policy::StageThreads,row=vec/8;
            const index_t ko=vec%8*16^Swizzle(row);
            const index_t offset=row*static_cast<index_t>(a.stride_As[0])+kb+ko;
            hcu_async_buffer_load_asm<int8_t,16>(dst+vec*16,ar,offset,0,
                P::Aligned || (mb+row<a.M && kb+ko+16<=a.K));
        });
    }
    CK_TILE_DEVICE BTile LoadB(const UniversalGemmKernelArgs<>& a,index_t nb,index_t kb,
                              const int32x4_t& br) const
    {
        const index_t lane=get_thread_id()%64,wave=get_thread_id()/64;
        BTile tile{};
        static_for<0,2,1>{}([&](auto half){
            constexpr index_t h=decltype(half)::value;
            static_for<0,P::NR,1>{}([&](auto j){
                constexpr index_t ni=decltype(j)::value;
                const index_t row=wave*(P::BN/P::WN)+ni*16+lane%16;
                const index_t ko=h*64+lane/16*16;
                const index_t offset=row*static_cast<index_t>(a.stride_Bs[0])+kb+ko;
                if constexpr(P::Aligned)
                    tile.values[h][ni]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4,false>(br,offset,0,true));
                else
                    tile.values[h][ni]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4>(br,offset,0,nb+row<a.N && kb+ko+16<=a.K));
            });
        });
        return tile;
    }
    CK_TILE_DEVICE void Compute(const int8_t* src,const BTile& b,Acc (&c)[P::MR][P::NR]) const
    {
        const index_t lane=get_thread_id()%64;
        static_for<0,2,1>{}([&](auto half){
            constexpr index_t h=decltype(half)::value;
            asm volatile("" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
            Copy av[P::MR];
            static_for<0,P::MR,1>{}([&](auto i){
                constexpr index_t mi=decltype(i)::value;const index_t row=mi*16+lane%16;
                if constexpr(P::LifetimeMode==1)
                    av[mi]=*reinterpret_cast<const volatile Copy*>(src+row*128+((h*64+lane/16*16)^Swizzle(row)));
                else
                    av[mi]=*reinterpret_cast<const Copy*>(src+row*128+((h*64+lane/16*16)^Swizzle(row)));
            });
            static_for<0,2,1>{}([&](auto kk){
                constexpr index_t ki=decltype(kk)::value;
                static_for<0,P::NR,1>{}([&](auto j){
                    constexpr index_t ni=decltype(j)::value;
                    auto bb=__builtin_shufflevector(b.values[h][ni],b.values[h][ni],ki*2,ki*2+1);
                    static_for<0,P::MR,1>{}([&](auto i){
                        constexpr index_t mi=decltype(i)::value;
                        auto aa=__builtin_shufflevector(av[mi],av[mi],ki*2,ki*2+1);
                        typename P::Mmac{}(c[mi][ni],aa,bb);
                    });
                });
            });
            asm volatile("" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
        });
    }
};
template<class P> struct GroupedInt8NtBypassBLifetimeProducer
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,int8_t* smem) const
    {
        GroupedInt8NtBypassBLifetimePipeline<P> pipe;
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const long long bytes=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(bytes<0x7fffffffLL?bytes:0x7fffffffLL));
        pipe.StageA(a,mb,0,smem,ar);asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
        index_t next_slot=1;
        for(index_t kb=128;kb<a.K;kb+=128)
        {
            pipe.StageA(a,mb,kb,smem+next_slot*32768,ar);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            // Next slot is ready and consumers have retired the previous one.
            block_sync_lds();next_slot^=1;
        }
    }
};
template<class P> struct GroupedInt8NtBypassBLifetimeConsumer
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem) const
    {
        __builtin_assume(get_thread_id()<P::ComputeThreads);
        GroupedInt8NtBypassBLifetimePipeline<P> pipe;
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
        GroupedInt8NtNoLdsEpilogue<P,true>{}(a,mb,nb,accum);
    }
};
}
