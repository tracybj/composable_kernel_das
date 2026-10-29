// SPDX-License-Identifier: MIT
#pragma once
#include "epilogue.hpp"

namespace ck_tile {
template<class P> struct GroupedInt8NtNReuseKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto o = args[blockIdx.z].group_karg;
        if(o.M <= 0 || o.N <= 0 || o.K <= 0) return;
        const UniversalGemmKernelArgs<> a{
            P::Transpose ? o.bs_ptr : o.as_ptr, P::Transpose ? o.as_ptr : o.bs_ptr, o.ds_ptr, o.e_ptr,
            P::Transpose ? o.N : o.M, P::Transpose ? o.M : o.N, o.K,
            P::Transpose ? o.stride_Bs : o.stride_As, P::Transpose ? o.stride_As : o.stride_Bs,
            o.stride_Ds, o.stride_E, o.k_batch};
        const index_t nt = (a.N + 255) / 256;
        const index_t cluster = blockIdx.x / (P::MGroup * nt);
        const index_t inner = blockIdx.x % (P::MGroup * nt);
        const index_t mb = (cluster * P::MGroup + inner % P::MGroup) * 128;
        const index_t nb = (inner / P::MGroup) * 256;
        if(mb >= a.M || nb >= a.N) return;
        __shared__ __align__(16) int8_t smem[16384];
        const bool native = a.M % 128 == 0 && a.N % 256 == 0 && a.K % 64 == 0 &&
            a.stride_Bs[0] > 0 && a.stride_Bs[0] < 16384 &&
            a.stride_As[0] % 16 == 0 && a.stride_Bs[0] % 16 == 0 &&
            (reinterpret_cast<uintptr_t>(a.as_ptr[0]) & 15u) == 0 &&
            (reinterpret_cast<uintptr_t>(a.bs_ptr[0]) & 15u) == 0 &&
            128LL * a.stride_As[0] <= 0x7fffffffLL &&
            static_cast<long long>(a.N - nb) * a.stride_Bs[0] <= 0x7fffffffLL;
        if(!native)
        {
            // Diagnostic fallback preserves tails/padded strides without entering the native protocol.
            const index_t valid_m=min(index_t(128),a.M-mb),valid_n=min(index_t(256),a.N-nb);
            for(index_t out=get_thread_id();out<valid_m*valid_n;out+=P::Threads){
                index_t row=mb+out/valid_n,col=nb+out%valid_n;int32_t sum=0;
                const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(row)*a.stride_As[0];
                const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(col)*a.stride_Bs[0];
                for(index_t k=0;k<a.K;++k)sum+=int32_t(ap[k])*int32_t(bp[k]);
                auto pos=P::Transpose?static_cast<long long>(col)*a.stride_E+row:static_cast<long long>(row)*a.stride_E+col;
                static_cast<int32_t*>(a.e_ptr)[pos]=sum;
            }
            return;
        }
        typename P::Pipeline::AccTile acc;
        const auto* ap = static_cast<const int8_t*>(a.as_ptr[0]) + static_cast<long long>(mb) * a.stride_As[0];
        const auto* bp = static_cast<const int8_t*>(a.bs_ptr[0]) + static_cast<long long>(nb) * a.stride_Bs[0];
        typename P::Pipeline{}.run_raw_into(ap, a.stride_As[0], bp, a.stride_Bs[0], a.N - nb,
                                          a.K / 64, acc, smem);
        GroupedInt8NtNReuseEpilogue<P>{}(a, mb, nb, acc);
    }
};
} // namespace ck_tile
