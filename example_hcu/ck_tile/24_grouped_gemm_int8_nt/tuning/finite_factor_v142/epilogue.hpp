// SPDX-License-Identifier: MIT
#pragma once
#include "../../cached_weights/kernels/scaled_epilogue.hpp"
namespace ck_tile {
// The factor, not each input scale, must be finite after its FP32 product.
// Integer accumulators are finite, so only this wave-uniform branch may use
// the RNE conversion without the generic NaN payload handling.
template <class P, bool BFloat>
struct GroupedInt8NtFiniteFactorEpilogue
{
    using Tile   = typename P::Pipeline::AccTile;
    using Parent = GroupedInt8NtScaledWaspEpilogue<P, BFloat>;
    CK_TILE_DEVICE static uint16_t finite_convert(int32_t acc, float factor)
    {
        float result = static_cast<float>(acc) * factor;
        asm volatile("" : "+v"(result));
        const uint32_t bits = bit_cast<uint32_t>(result);
        return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        if constexpr(!BFloat || P::Transpose)
        {
            Parent{}(a, mb, nb, tile, scale);
        }
        else
        {
            if(scale.a_stride != 0)
            {
                Parent{}(a, mb, nb, tile, scale);
                return;
            }
            const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
            if(wave < P::ProducerWaves)
                return;
            const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
            if constexpr(P::BN == 384)
                if(nb + consumer * 32 >= a.N)
                    return;
            const float sa = scale.a ? scale.a[0] : 1.0f;
            float factor[2][4];
            bool finite = true;
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 4, 1>{}([&](auto k) {
                    const index_t col =
                        nb + consumer * 32 + decltype(j)::value * 16 + h * 4 + decltype(k)::value;
                    const float f = sa * (scale.b ? scale.b[col * scale.b_stride] : 1.0f);
                    factor[decltype(j)::value][decltype(k)::value] = f;
                    finite = finite && (bit_cast<uint32_t>(f) & 0x7fffffffU) < 0x7f800000U;
                });
            });
            // All lanes must select the same path before the butterfly exchanges.
            if(__ballot(!finite) != 0)
            {
                Parent{}(a, mb, nb, tile, scale);
                return;
            }
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t row = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    const bool b0 = h & 1, b1 = h & 2;
                    int32_t t0    = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1);
                    int32_t t1    = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
                    r0            = b0 ? t0 : r0;
                    r1            = b0 ? r1 : t0;
                    r2            = b0 ? t1 : r2;
                    r3            = b0 ? r3 : t1;
                    t0            = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r2);
                    t1            = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r1 : r3);
                    r0            = b1 ? t0 : r0;
                    r2            = b1 ? r2 : t0;
                    r1            = b1 ? t1 : r1;
                    r3            = b1 ? r3 : t1;
                    const auto* f = factor[decltype(j)::value];
                    using V       = uint16_t __attribute__((ext_vector_type(4)));
                    const V out{finite_convert(r0, f[0]),
                                finite_convert(r1, f[1]),
                                finite_convert(r2, f[2]),
                                finite_convert(r3, f[3])};
                    const index_t col = nb + consumer * 32 + decltype(j)::value * 16 + h * 4;
                    *reinterpret_cast<V*>(static_cast<uint16_t*>(a.e_ptr) +
                                          static_cast<long long>(row) * a.stride_E + col) = out;
                });
            });
        }
    }
};
} // namespace ck_tile
