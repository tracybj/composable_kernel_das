// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
namespace ck_tile {
template<class P,class Policy=GroupedInt8NtD2LBankPolicy<P>>
struct GroupedInt8NtD2LBankPipeline
{
    using Mmac=typename P::Mmac;
    using Copy=typename P::CopyVector;
    using Acc=typename Mmac::CVecType;
    CK_TILE_DEVICE void Stage(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,index_t kb,
        int8_t* smem,const int32x4_t& ar,const int32x4_t& br) const
    {
        const index_t tid=get_thread_id();
        if constexpr(P::ProducerWaves>0) if(tid<P::ComputeThreads)return;
        const index_t stid=P::ProducerWaves?tid-P::ComputeThreads:tid;
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const bool native_a=static_cast<long long>(P::BM-1)*a.stride_As[0]+a.K<=0x7fffffffLL;
        const bool native_b=static_cast<long long>(P::BN-1)*a.stride_Bs[0]+a.K<=0x7fffffffLL;
        static_for<0,Policy::Issues,1>{}([&](auto issue){
            const index_t vec=stid+decltype(issue)::value*Policy::StageThreads;
            if(vec<Policy::Vectors)
            {
                if(vec<P::BM*4)
                {
                    const index_t row=vec/4, kvec=(vec%4)^((row>>P::SwizzleShift)&3);
                    const long long offset=static_cast<long long>(row)*a.stride_As[0]+kb+kvec*16;
                    const bool valid=mb+row<a.M && kb+kvec*16+16<=a.K;
                    if(native_a)
                        hcu_async_buffer_load_asm<int8_t,16>(smem+vec*16,ar,
                            static_cast<index_t>(offset),0,valid);
                    else
                    {
                        Copy v{};if(valid)v=*reinterpret_cast<const Copy*>(ap+offset);
                        *reinterpret_cast<Copy*>(smem+vec*16)=v;
                    }
                }
                else
                {
                    const index_t row=(vec-P::BM*4)/4,kvec=(vec%4)^((row>>P::SwizzleShift)&3);
                    const long long offset=static_cast<long long>(row)*a.stride_Bs[0]+kb+kvec*16;
                    const bool valid=nb+row<a.N && kb+kvec*16+16<=a.K;
                    if(native_b)
                        hcu_async_buffer_load_asm<int8_t,16>(smem+vec*16,br,
                            static_cast<index_t>(offset),0,valid);
                    else
                    {
                        Copy v{};if(valid)v=*reinterpret_cast<const Copy*>(bp+offset);
                        *reinterpret_cast<Copy*>(smem+vec*16)=v;
                    }
                }
            }
        });
    }
    CK_TILE_DEVICE void Compute(const int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const index_t tid=get_thread_id();
        if constexpr(P::ProducerWaves>0)if(tid>=P::ComputeThreads)return;
        const index_t lane=tid%64,wave=tid/64,wm=wave/P::WN,wn=wave%P::WN;
        Copy av[P::MR],bv[P::NR];
        static_for<0,P::MR,1>{}([&](auto i){
            constexpr int mi=decltype(i)::value;
            const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;
            const index_t ko=((lane/16)^((row>>P::SwizzleShift)&3))*16;
            av[mi]=*reinterpret_cast<const Copy*>(smem+row*64+ko);
        });
        static_for<0,P::NR,1>{}([&](auto j){
            constexpr int ni=decltype(j)::value;
            const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;
            const index_t ko=((lane/16)^((row>>P::SwizzleShift)&3))*16;
            bv[ni]=*reinterpret_cast<const Copy*>(smem+P::BM*64+row*64+ko);
        });
        // Both operands use the same K permutation: low/high 8-byte halves
        // cover all K64 entries exactly once across the two MMAC instructions.
        static_for<0,2,1>{}([&](auto kk){
            constexpr int ki=decltype(kk)::value;
            static_for<0,P::MR,1>{}([&](auto i){
                constexpr int mi=decltype(i)::value;
                auto aa=__builtin_shufflevector(av[mi],av[mi],ki*2,ki*2+1);
                static_for<0,P::NR,1>{}([&](auto j){
                    constexpr int ni=decltype(j)::value;
                    auto bb=__builtin_shufflevector(bv[ni],bv[ni],ki*2,ki*2+1);
                    Mmac{}(c[mi][ni],aa,bb);
                });
            });
        });
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,
        int8_t* smem,Acc (&c)[P::MR][P::NR]) const
    {
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const long long a_bytes=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K;
        const long long b_bytes=static_cast<long long>(a.N-nb-1)*a.stride_Bs[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(a_bytes<0x7fffffffLL?a_bytes:0x7fffffffLL));
        const auto br=make_wave_buffer_resource(bp,static_cast<uint32_t>(b_bytes<0x7fffffffLL?b_bytes:0x7fffffffLL));
        if constexpr(P::DoubleBuffer)
        {
            Stage(a,mb,nb,0,smem,ar,br);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
            index_t buffer=0;
            for(index_t kb=0;kb<a.K;kb+=64)
            {
                if(kb+64<a.K) Stage(a,mb,nb,kb+64,smem+(1-buffer)*Policy::InputBytes,ar,br);
                Compute(smem+buffer*Policy::InputBytes,c);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
                buffer=1-buffer;
            }
        }
        else
        {
            for(index_t kb=0;kb<a.K;kb+=64)
            {
                Stage(a,mb,nb,kb,smem,ar,br);
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
                Compute(smem,c);block_sync_lds();
            }
        }
    }
};
}
