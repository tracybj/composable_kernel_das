// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
#include "../transposed_bypass_v54/epilogue.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtStructBProducer
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,int8_t* smem) const
    {
        using Policy=GroupedInt8NtStructBPolicy<P>;
        const index_t stid=get_thread_id()-P::ComputeThreads;
        __builtin_assume(stid>=0 && stid<256);
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const long long bytes=static_cast<long long>(a.M-mb-1)*a.stride_As[0]+a.K;
        const auto ar=make_wave_buffer_resource(ap,static_cast<uint32_t>(bytes<0x7fffffffLL?bytes:0x7fffffffLL));
        const index_t first_vec=P::SlabCopy?(stid/64)*512+stid%64:stid;
        const uint32_t wave_base=__builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(smem+first_vec*16));
        typename Policy::Offsets offsets;
        static_for<0,8,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;
            const index_t vec=P::SlabCopy?(stid/64)*512+j*64+stid%64:stid+j*256;
            const index_t row=vec/8,ko=vec%8*16^((row&7)*16);
            offsets.v[j]=row*static_cast<index_t>(a.stride_As[0])+ko;
            // Preserve the precomputed value across loops instead of rematerializing it.
            asm volatile("" : "+v"(offsets.v[j]));
        });
        for(index_t kb=0;kb<a.K;kb+=128)
        {
            Policy::CopyA(wave_base+((kb/128)&1)*32768,ar,offsets,kb);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
        }
    }
};
template<class P> struct GroupedInt8NtStructBConsumer
{
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,int8_t* smem) const
    {
        using Policy=GroupedInt8NtStructBPolicy<P>;
        const index_t tid=get_thread_id();__builtin_assume(tid<P::ComputeThreads);
        const index_t lane=tid%64,wave=tid/64;
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        const auto br=Policy::BResource(bp,a.stride_Bs[0],static_cast<uint32_t>(static_cast<long long>(a.N-nb)*a.stride_Bs[0]));
        const uint32_t row=wave*32+lane%16,ko=lane/16*16;
        uint64_t p0=(static_cast<uint64_t>(ko)<<32)|row,p1=(static_cast<uint64_t>(ko)<<32)|(row+16u);
        asm volatile("" : "+v"(p0),"+v"(p1));
        typename P::Mmac::CVecType accum[P::MR][P::NR]{};
        GroupedInt8NtBypassBOrderedPipeline<P> compute;
        auto current=Policy::LoadB(br,p0,p1,0);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
        index_t slot=0;
        for(index_t kb=0;kb+128<a.K;kb+=128)
        {
            auto next=Policy::LoadB(br,p0,p1,kb+128);
            compute.Compute(smem+slot*32768,current,accum);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");block_sync_lds();
            current=next;slot^=1;
        }
        compute.Compute(smem+slot*32768,current,accum);
        if constexpr(P::Transpose)GroupedInt8NtTransposedEpilogue<P>{}(a,mb,nb,accum);
        else GroupedInt8NtNoLdsEpilogue<P,true>{}(a,mb,nb,accum);
    }
};
}
