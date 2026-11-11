// SPDX-License-Identifier: MIT
#pragma once
#include "../../kernels/scaled_epilogue.hpp"

namespace ck_tile {
// Keep the same arithmetic/rounding contract. Exchange packed16-bit output
// values instead of32-bit accumulators,using three wave exchanges per quartet.
// Only the native original orientation/scalar-A path is changed by this policy.
template <class P, bool BFloat>
struct GroupedInt8NtPackedOutputEpilogue
{
    using Tile   = typename P::Pipeline::AccTile;
    using Parent = GroupedInt8NtScaledWaspEpilogue<P, BFloat>;

    template <bool ChannelB>
    CK_TILE_DEVICE void scalar_a(const UniversalGemmKernelArgs<>& original,
                                 index_t mb,
                                 index_t nb,
                                 const Tile& tile,
                                 Int8NtScaleArgs scale,
                                 float sa,
                                 float sb) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        if(wave < P::ProducerWaves)
            return;
        const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
        if constexpr(P::BN == 384)
            if(nb + consumer * 32 >= original.N)
                return;
        float factor[2][4];
        static_for<0, 2, 1>{}([&](auto j) {
            static_for<0, 4, 1>{}([&](auto k) {
                const index_t col =
                    nb + consumer * 32 + decltype(j)::value * 16 + h + decltype(k)::value * 4;
                factor[decltype(j)::value][decltype(k)::value] =
                    sa * (ChannelB ? scale.b[col] : sb);
            });
        });
        static_for<0, P::BM / 16, 1>{}([&](auto i) {
            const index_t row = mb + decltype(i)::value * 16 + lane % 16;
            static_for<0, 2, 1>{}([&](auto j) {
                const auto v      = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                const auto* f     = factor[decltype(j)::value];
                const uint32_t x0 = Int8NtDequantConvert<BFloat>::run(v[0], f[0]);
                const uint32_t x1 = Int8NtDequantConvert<BFloat>::run(v[1], f[1]);
                const uint32_t x2 = Int8NtDequantConvert<BFloat>::run(v[2], f[2]);
                const uint32_t x3 = Int8NtDequantConvert<BFloat>::run(v[3], f[3]);
                uint32_t r0 = x0 | (x1 << 16), r1 = x2 | (x3 << 16);
                const bool b0 = h & 1, b1 = h & 2;
                const uint32_t t0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, r0);
                const uint32_t t1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, r1);
                r0 = b0 ? ((t0 >> 16) | (r0 & 0xffff0000U)) : ((r0 & 0xffffU) | (t0 << 16));
                r1 = b0 ? ((t1 >> 16) | (r1 & 0xffff0000U)) : ((r1 & 0xffffU) | (t1 << 16));
                const uint32_t t = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r1);
                using V          = ext_vector_t<uint32_t, 2>;
                const V out{b1 ? t : r0, b1 ? r1 : t};
                const index_t col = nb + consumer * 32 + decltype(j)::value * 16 + h * 4;
                *reinterpret_cast<V*>(static_cast<uint16_t*>(original.e_ptr) +
                                      static_cast<long long>(row) * original.stride_E + col) = out;
            });
        });
    }

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& original,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        if constexpr(P::Transpose)
            Parent{}(original, mb, nb, tile, scale);
        else
        {
            if(scale.a_stride != 0)
            {
                Parent{}(original, mb, nb, tile, scale);
                return;
            }
            const float sa = scale.a ? scale.a[0] : 1.0f;
            if(scale.b && scale.b_stride == 1)
                scalar_a<true>(original, mb, nb, tile, scale, sa, 1.0f);
            else
                scalar_a<false>(original, mb, nb, tile, scale, sa, scale.b ? scale.b[0] : 1.0f);
        }
    }
};
} // namespace ck_tile
