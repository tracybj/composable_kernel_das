// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
namespace ck_tile {
template<class P,class Policy=GroupedInt8NtStreamPolicy<P>> struct GroupedInt8NtStreamPipeline
{
    using Mmac=typename P::Mmac;using Acc=typename Mmac::CVecType;
    CK_TILE_DEVICE static void WaitKeep()
    {
        if constexpr(Policy::Pending==0)asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        else if constexpr(Policy::Pending==2)asm volatile("s_waitcnt vmcnt(2)" ::: "memory");
        else if constexpr(Policy::Pending==4)asm volatile("s_waitcnt vmcnt(4)" ::: "memory");
        else if constexpr(Policy::Pending==8)asm volatile("s_waitcnt vmcnt(8)" ::: "memory");
        else static_assert(Policy::Pending==0,"Unsupported FIFO threshold");
    }
    CK_TILE_DEVICE void Stage(const UniversalGemmKernelArgs<>& a,index_t kb,int8_t* smem,
                              const int32x4_t& ar,const int32x4_t& br) const
    {
        const index_t tid=get_thread_id();
        if constexpr(P::ProducerWaves>0)if(!P::IsProducer(tid))return;
        const index_t stid=P::StageThread(tid);
        static_for<0,Policy::Issues,1>{}([&](auto i){
            const index_t vec=stid+decltype(i)::value*Policy::StageThreads;
            const bool is_a=vec<P::BM*2;
            const index_t row=(is_a?vec:vec-P::BM*2)/2;
            const index_t kv=(vec%2)^((row>>P::SwizzleShift)&1);
            // The outer native-span guard proves this nonnegative sum fits i32.
            const index_t offset=row*static_cast<index_t>(is_a?a.stride_As[0]:a.stride_Bs[0])+kb+kv*16;
            if(is_a)hcu_async_buffer_load_asm<int8_t,16>(smem+vec*16,ar,offset,0,true);
            else hcu_async_buffer_load_asm<int8_t,16>(smem+vec*16,br,offset,0,true);
        });
    }
    CK_TILE_DEVICE void Compute(const int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const index_t tid=get_thread_id(),wave=P::ComputeWave(tid);
        if(wave<0)return;
        const index_t lane=tid%64,wm=wave/P::WN,wn=wave%P::WN;
        typename Mmac::AVecType av[P::MR];typename Mmac::BVecType bv[P::NR];
        static_for<0,P::MR,1>{}([&](auto i){
            constexpr index_t mi=decltype(i)::value;
            const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;
            const index_t ko=(lane/16*8)^(((row>>P::SwizzleShift)&1)*16);
            av[mi]=*reinterpret_cast<const typename Mmac::AVecType*>(smem+row*32+ko);
        });
        static_for<0,P::NR,1>{}([&](auto j){
            constexpr index_t ni=decltype(j)::value;
            const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;
            const index_t ko=(lane/16*8)^(((row>>P::SwizzleShift)&1)*16);
            bv[ni]=*reinterpret_cast<const typename Mmac::BVecType*>(smem+P::BM*32+row*32+ko);
        });
        if constexpr(P::Priority)asm volatile("s_setprio 1" ::: "memory");
        static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){
            Mmac{}(c[decltype(i)::value][decltype(j)::value],av[decltype(i)::value],bv[decltype(j)::value]);
        });});
        if constexpr(P::Priority)asm volatile("s_setprio 0" ::: "memory");
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,
                                   int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const long long ab=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K;
        const long long bb=static_cast<long long>(a.N-nb-1)*a.stride_Bs[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(ab<0x7fffffffLL?ab:0x7fffffffLL));
        const auto br=make_wave_buffer_resource(bp,static_cast<uint32_t>(bb<0x7fffffffLL?bb:0x7fffffffLL));
        static_for<0,P::Stages-1,1>{}([&](auto i){
            Stage(a,decltype(i)::value*32,smem+decltype(i)::value*Policy::InputBytes,ar,br);
        });
        WaitKeep();block_sync_lds();
        index_t read_slot=0,write_slot=P::Stages-1;
        for(index_t kb=0;kb<a.K;kb+=32)
        {
            const bool future=kb+(P::Stages-1)*32<a.K;
            if(future)Stage(a,kb+(P::Stages-1)*32,smem+write_slot*Policy::InputBytes,ar,br);
            Compute(smem+read_slot*Policy::InputBytes,c);
            // Retire the next readable tile; retain later copies in flight.
            if(future)WaitKeep();else asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            block_sync_lds();
            read_slot=read_slot+1==P::Stages?0:read_slot+1;
            write_slot=write_slot+1==P::Stages?0:write_slot+1;
        }
    }
};
}
