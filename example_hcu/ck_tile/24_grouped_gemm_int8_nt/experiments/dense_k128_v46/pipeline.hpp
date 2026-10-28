// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
namespace ck_tile {
template<class P,class Policy=GroupedInt8NtDenseKPolicy<P>> struct GroupedInt8NtDenseKPipeline
{
    using Mmac=typename P::Mmac;using Copy=typename P::CopyVector;using Acc=typename Mmac::CVecType;
    struct GlobalTile {Copy a[Policy::AVectors],b[Policy::BVectors];};
    CK_TILE_DEVICE static bool Producer(index_t tid){return P::ProducerWaves>0 && tid>=P::ComputeThreads;}
    CK_TILE_DEVICE static index_t StageThread(index_t tid){return P::ProducerWaves?tid-P::ComputeThreads:tid;}
    CK_TILE_DEVICE static index_t Swizzle(index_t row){return P::Swizzle8?(row&15)*8:(row&7)*16;}
    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,index_t kb,
                                  const int32x4_t& ar,const int32x4_t& br) const
    {
        GlobalTile tile{};const index_t tid=get_thread_id();
        if constexpr(P::ProducerWaves>0)if(!Producer(tid))return tile;
        const index_t stid=StageThread(tid);
        static_for<0,Policy::AVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;
            const index_t vec=stid+j*Policy::StageThreads,row=vec/8,ko=vec%8*16;
            const index_t offset=row*static_cast<index_t>(a.stride_As[0])+kb+ko;
            if constexpr(P::Aligned)tile.a[j]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4,false>(ar,offset,0,true));
            else tile.a[j]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4>(ar,offset,0,mb+row<a.M && kb+ko+16<=a.K));
        });
        static_for<0,Policy::BVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;
            const index_t vec=stid+j*Policy::StageThreads,row=vec/8,ko=vec%8*16;
            const index_t offset=row*static_cast<index_t>(a.stride_Bs[0])+kb+ko;
            if constexpr(P::Aligned)tile.b[j]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4,false>(br,offset,0,true));
            else tile.b[j]=bit_cast<Copy>(hcu_buffer_load_asm<int32_t,4>(br,offset,0,nb+row<a.N && kb+ko+16<=a.K));
        });
        return tile;
    }
    CK_TILE_DEVICE static void Write(int8_t* dst,index_t row,index_t ko,const Copy& v)
    {
        if constexpr(P::Swizzle8)
        {
            auto lo=__builtin_shufflevector(v,v,0,1),hi=__builtin_shufflevector(v,v,2,3);
            *reinterpret_cast<decltype(lo)*>(dst+row*128+(ko^Swizzle(row)))=lo;
            *reinterpret_cast<decltype(hi)*>(dst+row*128+((ko+8)^Swizzle(row)))=hi;
        }
        else *reinterpret_cast<Copy*>(dst+row*128+(ko^Swizzle(row)))=v;
    }
    CK_TILE_DEVICE void Stage(const GlobalTile& tile,int8_t* smem) const
    {
        const index_t tid=get_thread_id();if constexpr(P::ProducerWaves>0)if(!Producer(tid))return;
        const index_t stid=StageThread(tid);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        static_for<0,Policy::AVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;const index_t vec=stid+j*Policy::StageThreads;
            Write(smem,vec/8,vec%8*16,tile.a[j]);
        });
        static_for<0,Policy::BVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;const index_t vec=stid+j*Policy::StageThreads;
            Write(smem+Policy::BPitchOffset,vec/8,vec%8*16,tile.b[j]);
        });
    }
    CK_TILE_DEVICE void Compute(const int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const index_t tid=get_thread_id();if constexpr(P::ProducerWaves>0)if(Producer(tid))return;
        const index_t lane=tid%64,wave=tid/64,wm=wave/P::WN,wn=wave%P::WN;
        if constexpr(P::Paired)
        {
            static_for<0,2,1>{}([&](auto half){
                constexpr index_t h=decltype(half)::value;
                if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);
                Copy av[P::MR],bv[P::NR];
                static_for<0,P::MR,1>{}([&](auto i){
                    constexpr index_t mi=decltype(i)::value;const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;
                    av[mi]=*reinterpret_cast<const Copy*>(smem+row*128+((h*64+lane/16*16)^Swizzle(row)));
                });
                static_for<0,P::NR,1>{}([&](auto j){
                    constexpr index_t ni=decltype(j)::value;const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;
                    bv[ni]=*reinterpret_cast<const Copy*>(smem+Policy::BPitchOffset+row*128+((h*64+lane/16*16)^Swizzle(row)));
                });
                static_for<0,2,1>{}([&](auto kk){
                    constexpr int ki=decltype(kk)::value;
                    static_for<0,P::MR,1>{}([&](auto i){
                        constexpr int mi=decltype(i)::value;auto aa=__builtin_shufflevector(av[mi],av[mi],ki*2,ki*2+1);
                        static_for<0,P::NR,1>{}([&](auto j){
                            constexpr int ni=decltype(j)::value;auto bb=__builtin_shufflevector(bv[ni],bv[ni],ki*2,ki*2+1);
                            Mmac{}(c[mi][ni],aa,bb);
                        });
                    });
                });
                if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);
            });
        }
        else
        {
            static_for<0,4,1>{}([&](auto kk){
                constexpr index_t ki=decltype(kk)::value;
                if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);
                typename Mmac::AVecType av[P::MR];typename Mmac::BVecType bv[P::NR];
                static_for<0,P::MR,1>{}([&](auto i){
                    constexpr index_t mi=decltype(i)::value;const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;
                    av[mi]=*reinterpret_cast<const typename Mmac::AVecType*>(smem+row*128+((ki*32+lane/16*8)^Swizzle(row)));
                });
                static_for<0,P::NR,1>{}([&](auto j){
                    constexpr index_t ni=decltype(j)::value;const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;
                    bv[ni]=*reinterpret_cast<const typename Mmac::BVecType*>(smem+Policy::BPitchOffset+row*128+((ki*32+lane/16*8)^Swizzle(row)));
                });
                static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){
                    Mmac{}(c[decltype(i)::value][decltype(j)::value],av[decltype(i)::value],bv[decltype(j)::value]);
                });});
                if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);
            });
        }
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const long long ab=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K,bb=static_cast<long long>(a.N-nb-1)*a.stride_Bs[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(ab<0x7fffffffLL?ab:0x7fffffffLL));
        const auto br=make_wave_buffer_resource(bp,static_cast<uint32_t>(bb<0x7fffffffLL?bb:0x7fffffffLL));
        auto current=Load(a,mb,nb,0,ar,br);Stage(current,smem);block_sync_lds();
        for(index_t kb=0;kb+128<a.K;kb+=128)
        {
            auto next=Load(a,mb,nb,kb+128,ar,br);
            __builtin_amdgcn_sched_barrier(0);Compute(smem,c);__builtin_amdgcn_sched_barrier(0);
            block_sync_lds();Stage(next,smem);block_sync_lds();
        }
        Compute(smem,c);
    }
};
}
