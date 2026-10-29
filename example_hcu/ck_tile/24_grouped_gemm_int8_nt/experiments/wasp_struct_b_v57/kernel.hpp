// SPDX-License-Identifier: MIT
#pragma once
#include "pipeline.hpp"
#include "../dense_k128_v46/kernel.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtStructBKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto o=args[blockIdx.z].group_karg;
        if(o.M<=0 || o.N<=0 || o.K<=0)return;
        const UniversalGemmKernelArgs<> a{
            P::Transpose?o.bs_ptr:o.as_ptr,P::Transpose?o.as_ptr:o.bs_ptr,o.ds_ptr,o.e_ptr,
            P::Transpose?o.N:o.M,P::Transpose?o.M:o.N,o.K,
            P::Transpose?o.stride_Bs:o.stride_As,P::Transpose?o.stride_As:o.stride_Bs,
            o.stride_Ds,o.stride_E,o.k_batch};
        const index_t nt=(a.N+P::BN-1)/P::BN,cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM,nb=(inner/P::MGroup)*P::BN;
        if(mb>=a.M || nb>=a.N)return;
        __shared__ __align__(16) int8_t smem[65536];
        const long long padded_k=(static_cast<long long>(a.K)+127)/128*128;
        const bool offsets=255LL*a.stride_As[0]+padded_k<=0x7fffffffLL && 255LL*a.stride_Bs[0]+padded_k<=0x7fffffffLL;
        const bool structured=a.stride_Bs[0]>0 && a.stride_Bs[0]<16384 && static_cast<long long>(a.N-nb)*a.stride_Bs[0]<=0x7fffffffLL;
        // Native structured loads cover complete K128/M256/N256 tiles. Other
        // requests retain the fully masked, 64-bit-stride generic fallback.
        if(!offsets || !structured || a.M%256 || a.N%256 || a.K%128)
        {
            using F=GroupedInt8NtDenseKFallback<P>;
            typename F::Mmac::CVecType acc[F::MR][F::NR]{};
            GroupedInt8NtD2LGeometryPipeline<F>{}(a,mb,nb,smem,acc);
            if constexpr(P::Transpose)GroupedInt8NtTransposedEpilogue<F>{}(a,mb,nb,acc);
            else GroupedInt8NtNoLdsEpilogue<F,true>{}(a,mb,nb,acc);
            return;
        }
        const index_t wave=__builtin_amdgcn_readfirstlane(get_thread_id()/64);
        if(wave>=8)GroupedInt8NtStructBProducer<P>{}(a,mb,smem);
        else GroupedInt8NtStructBConsumer<P>{}(a,mb,nb,smem);
    }
};
}
