// SPDX-License-Identifier: MIT
#pragma once
#include "../../kernels/grouped_int8_nt_scaled_epilogue.hpp"
namespace ck_tile {
// Experimental scalar-A BF16 path: preserve conversion/scale rounding, then
// exchange two packed16 lanes at once. No change to the GEMM main loop.
template <class P, bool BFloat>
struct GroupedInt8NtPacked16ExchangeEpilogue
{
    using Tile = typename P::Pipeline::AccTile;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        if constexpr(!BFloat || P::Transpose)
        {
            GroupedInt8NtScaledWaspEpilogue<P, BFloat>{}(a, mb, nb, tile, scale);
        }
        else
        {
            if(scale.a_stride != 0)
            {
                GroupedInt8NtScaledWaspEpilogue<P, BFloat>{}(a, mb, nb, tile, scale);
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
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 4, 1>{}([&](auto k) {
                    // These are the ORIGINAL accumulator columns before the lane transpose.
                    const index_t col =
                        nb + consumer * 32 + decltype(j)::value * 16 + h + decltype(k)::value * 4;
                    factor[decltype(j)::value][decltype(k)::value] =
                        sa * (scale.b ? scale.b[col * scale.b_stride] : 1.0f);
                });
            });
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t row = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v  = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    const auto* f = factor[decltype(j)::value];
                    uint32_t r0   = Int8NtDequantConvert<true>::run(v[0], f[0]);
                    uint32_t r1   = Int8NtDequantConvert<true>::run(v[1], f[1]);
                    uint32_t r2   = Int8NtDequantConvert<true>::run(v[2], f[2]);
                    uint32_t r3   = Int8NtDequantConvert<true>::run(v[3], f[3]);
                    const bool b0 = h & 1, b1 = h & 2;
                    uint32_t packed = (b0 ? r0 : r1) | ((b0 ? r2 : r3) << 16);
                    uint32_t t      = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, packed);
                    if(b0)
                    {
                        r0 = t & 65535u;
                        r2 = t >> 16;
                    }
                    else
                    {
                        r1 = t & 65535u;
                        r3 = t >> 16;
                    }
                    packed = (b1 ? r0 : r2) | ((b1 ? r1 : r3) << 16);
                    t      = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, packed);
                    if(b1)
                    {
                        r0 = t & 65535u;
                        r1 = t >> 16;
                    }
                    else
                    {
                        r2 = t & 65535u;
                        r3 = t >> 16;
                    }
                    using V = uint16_t __attribute__((ext_vector_type(4)));
                    const V out{static_cast<uint16_t>(r0),
                                static_cast<uint16_t>(r1),
                                static_cast<uint16_t>(r2),
                                static_cast<uint16_t>(r3)};
                    const index_t col = nb + consumer * 32 + decltype(j)::value * 16 + h * 4;
                    *reinterpret_cast<V*>(static_cast<uint16_t*>(a.e_ptr) +
                                          static_cast<long long>(row) * a.stride_E + col) = out;
                });
            });
        }
    }
};
} // namespace ck_tile
