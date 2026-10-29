// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "../transposed_bypass_v54/pipeline.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtHoistedCopyProducer
{
    using Policy=GroupedInt8NtHoistedCopyPolicy;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,int8_t* smem) const
    {
        static_assert(P::AShift==0,"Hoisted raw-copy offsets use the row&7 XOR layout");
        const index_t stid=get_thread_id()-P::ComputeThreads;
        __builtin_assume(stid>=0 && stid<256);
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const long long bytes=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K;
        auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(bytes<0x7fffffffLL?bytes:0x7fffffffLL));
        const uint32_t wave_base=__builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(smem+stid*16));
        typename Policy::Offsets offsets;
        static_for<0,8,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;const index_t vec=stid+j*256,row=vec/8;
            const index_t ko=vec%8*16^((row&7)*16);
            offsets.v[j]=row*static_cast<index_t>(a.stride_As[0])+ko;
        });
        for(index_t kb=0;kb<a.K;kb+=128)
        {
            typename Policy::Offsets active=offsets;
            static_for<0,8,1>{}([&](auto i){
                constexpr index_t j=decltype(i)::value;const index_t vec=stid+j*256,row=vec/8;
                const index_t ko=vec%8*16^((row&7)*16);
                const bool valid=P::Aligned || (mb+row<a.M && kb+ko+16<=a.K);
                const index_t offset=P::AdvanceResource?offsets.v[j]:offsets.v[j]+kb;
                active.v[j]=valid?offset:ar[2];
            });
            Policy::Copy(wave_base+((kb/128)&1)*32768,ar,active);
            // First publication; subsequent copies overlap consumer compute.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
            if constexpr(P::AdvanceResource)Policy::Advance(ar);
        }
    }
};
}
