// SPDX-License-Identifier: MIT
#pragma once
#include "epilogue.hpp"
#include "../../grouped_int8_nt_wasp_fallback.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtWaspWideKernel
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
        const index_t mb = (cluster * P::MGroup + inner % P::MGroup) * 512;
        const index_t nb = (inner / P::MGroup) * 256;
        if(mb >= a.M || nb >= a.N) return;
        __shared__ __align__(16) int8_t smem[65536];
        const bool native = a.M % 512 == 0 && a.N % 256 == 0 && a.K % 64 == 0 &&
            a.stride_Bs[0] > 0 && a.stride_Bs[0] < 16384 &&
            a.stride_As[0] % 16 == 0 && a.stride_Bs[0] % 16 == 0 &&
            (reinterpret_cast<uintptr_t>(a.as_ptr[0]) & 15u) == 0 &&
            (reinterpret_cast<uintptr_t>(a.bs_ptr[0]) & 15u) == 0 &&
            512LL * a.stride_As[0] <= 0x7fffffffLL &&
            static_cast<long long>(a.N - nb) * a.stride_Bs[0] <= 0x7fffffffLL;
        if(!native)
        {
            GroupedInt8NtWaspFallback<P>{}(a, mb, nb, smem);
            if(mb + 256 < a.M)
            {
                block_sync_lds();
                GroupedInt8NtWaspFallback<P>{}(a, mb + 256, nb, smem);
            }
            return;
        }
        typename P::Pipeline::AccTile acc;
        const auto* ap = static_cast<const int8_t*>(a.as_ptr[0]) + static_cast<long long>(mb) * a.stride_As[0];
        const auto* bp = static_cast<const int8_t*>(a.bs_ptr[0]) + static_cast<long long>(nb) * a.stride_Bs[0];
        typename P::Pipeline{}.run_raw_into(ap, a.stride_As[0], bp, a.stride_Bs[0], a.N - nb,
                                          a.K / 64, acc, smem);
        GroupedInt8NtWaspWideEpilogue<P>{}(a, mb, nb, acc);
    }
};
} // namespace ck_tile
