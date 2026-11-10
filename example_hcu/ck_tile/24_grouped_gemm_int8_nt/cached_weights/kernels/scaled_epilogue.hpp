// SPDX-License-Identifier: MIT
#pragma once
#include "../../kernels/grouped_int8_nt_policy.hpp"
#include "../scale_args.hpp"
namespace ck_tile {
// Keep the FP32 rounding boundary before the selected 16-bit conversion.
// Explicit select avoids a divergent NaN-handling region for every BF16 value.
template <bool BFloat>
struct Int8NtDequantConvert
{
    CK_TILE_DEVICE static uint16_t run(int32_t value, float combined_scale)
    {
        float result = static_cast<float>(value) * combined_scale;
        asm volatile("" : "+v"(result));
        if constexpr(BFloat)
        {
            const uint32_t bits    = bit_cast<uint32_t>(result);
            const uint32_t rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
            const uint32_t nan     = (bits >> 16) | 0x40U;
            uint32_t selected;
            asm volatile("v_cmp_lt_u32 vcc, %4, %1\n\t"
                         "v_cndmask_b32 %0, %2, %3, vcc"
                         : "=v"(selected)
                         : "v"(bits & 0x7fffffffU), "v"(rounded), "v"(nan), "s"(0x7f800000U)
                         : "vcc");
            return static_cast<uint16_t>(selected);
        }
        else
            return bit_cast<uint16_t>(type_convert<half_t>(result));
    }
};
// Scales use ORIGINAL output coordinates, including the logical-transpose route.
// mode strides are 0 for a scalar and 1 for an outer-vector scale.
template <bool BFloat>
struct Int8NtDequantStore
{
    Int8NtScaleArgs scale;
    CK_TILE_DEVICE uint16_t convert(int32_t value, index_t row, index_t col) const
    {
        const float sa = scale.a ? scale.a[row * scale.a_stride] : 1.0f;
        const float sb = scale.b ? scale.b[col * scale.b_stride] : 1.0f;
        // Match the installed BLASLt SAB epilogue:round the product of FP32
        // scales before applying it to the converted integer accumulator.
        return Int8NtDequantConvert<BFloat>::run(value, sa * sb);
    }
    CK_TILE_DEVICE void
    one(const UniversalGemmKernelArgs<>& a, index_t row, index_t col, int32_t value) const
    {
        static_cast<uint16_t*>(a.e_ptr)[static_cast<long long>(row) * a.stride_E + col] =
            convert(value, row, col);
    }
    CK_TILE_DEVICE void four(const UniversalGemmKernelArgs<>& a,
                             index_t row,
                             index_t col,
                             int32_t r0,
                             int32_t r1,
                             int32_t r2,
                             int32_t r3) const
    {
        using V        = uint16_t __attribute__((ext_vector_type(4)));
        const float sa = scale.a ? scale.a[row * scale.a_stride] : 1.0f;
        const float b0 = scale.b ? scale.b[col * scale.b_stride] : 1.0f;
        const float b1 = scale.b ? scale.b[(col + 1) * scale.b_stride] : 1.0f;
        const float b2 = scale.b ? scale.b[(col + 2) * scale.b_stride] : 1.0f;
        const float b3 = scale.b ? scale.b[(col + 3) * scale.b_stride] : 1.0f;
        const V v{Int8NtDequantConvert<BFloat>::run(r0, sa * b0),
                  Int8NtDequantConvert<BFloat>::run(r1, sa * b1),
                  Int8NtDequantConvert<BFloat>::run(r2, sa * b2),
                  Int8NtDequantConvert<BFloat>::run(r3, sa * b3)};
        *reinterpret_cast<V*>(static_cast<uint16_t*>(a.e_ptr) +
                              static_cast<long long>(row) * a.stride_E + col) = v;
    }
};
template <class P, bool BFloat>
struct GroupedInt8NtScaledTileEpilogue
{
    using Acc    = typename P::Mmac::CVecType;
    using Policy = GroupedInt8NtTilePolicy<P>;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   const Acc (&c)[P::MR][P::NR],
                                   Int8NtScaleArgs scale) const
    {
        const index_t tid = get_thread_id(), lane = tid % 64, wave = tid / 64, wm = wave / P::WN,
                      wn = wave % P::WN;
        auto* tmp        = reinterpret_cast<int32_t*>(smem);
        const Int8NtDequantStore<BFloat> store{scale};
        block_sync_lds();
        static_for<0, P::MR, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            if constexpr(P::BatchedEpi)
            {
                const index_t row = wm * 16 + lane % 16;
                static_for<0, P::NR, 1>{}([&](auto j) {
                    static_for<0, 4, 1>{}([&](auto v) {
                        constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                        const index_t col = wn * (P::BN / P::WN) + ni * 16 + lane / 16 + vi * 4;
                        tmp[row * Policy::CPitch + (col ^ ((row % 8) * 4))] = c[mi][ni][vi];
                    });
                });
                block_sync_lds();
                for(index_t vec = tid; vec < Policy::CStripeRows * P::BN / 4; vec += P::Threads)
                {
                    const index_t r = vec / (P::BN / 4), col = (vec % (P::BN / 4)) * 4;
                    const index_t rowout = mb + (r / 16) * (P::BM / P::WM) + mi * 16 + r % 16;
                    if(P::Aligned || (rowout < a.M && nb + col + 4 <= a.N))
                    {
                        const auto v = *reinterpret_cast<const typename P::CopyVector*>(
                            tmp + r * Policy::CPitch + (col ^ ((r % 8) * 4)));
                        store.four(a, rowout, nb + col, v[0], v[1], v[2], v[3]);
                    }
                }
                block_sync_lds();
            }
            else
            {
                static_for<0, P::WM, 1>{}([&](auto w) {
                    if(wm == decltype(w)::value)
                        static_for<0, P::NR, 1>{}([&](auto j) {
                            static_for<0, 4, 1>{}([&](auto v) {
                                constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                                const index_t col =
                                    wn * (P::BN / P::WN) + ni * 16 + lane / 16 + vi * 4;
                                tmp[(lane % 16) * Policy::CPitch + col] = c[mi][ni][vi];
                            });
                        });
                    block_sync_lds();
                    for(index_t vec = tid; vec < Policy::CStripeRows * P::BN / 4; vec += P::Threads)
                    {
                        const index_t r = vec / (P::BN / 4), col = (vec % (P::BN / 4)) * 4;
                        const index_t rowout =
                            mb + decltype(w)::value * (P::BM / P::WM) + mi * 16 + r;
                        if(P::Aligned || (rowout < a.M && nb + col + 4 <= a.N))
                        {
                            const auto v = *reinterpret_cast<const typename P::CopyVector*>(
                                tmp + r * Policy::CPitch + col);
                            store.four(a, rowout, nb + col, v[0], v[1], v[2], v[3]);
                        }
                    }
                    block_sync_lds();
                });
            }
        });
    }
};
template <class P, bool BFloat>
struct GroupedInt8NtScaledWaspEpilogue
{
    using Tile = typename P::Pipeline::AccTile;
    // The native WASP domain has complete tiles (or complete 32-column consumer
    // stripes for BN384). Its output bounds are proved once in the kernel.
    // A scalar activation scale and channel/scalar weight scales are reused
    // over the accumulator tile rather than loaded for each output value.
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
            if(nb + consumer * 32 >= (P::Transpose ? original.M : original.N))
                return;
        if constexpr(P::Transpose)
        {
            float factor[P::BM / 16];
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t col          = mb + decltype(i)::value * 16 + lane % 16;
                factor[decltype(i)::value] = sa * (ChannelB ? scale.b[col] : sb);
            });
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t col = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                    static_for<0, 4, 1>{}([&](auto k) {
                        const index_t row = base + h + decltype(k)::value * 4;
                        static_cast<uint16_t*>(
                            original.e_ptr)[static_cast<long long>(row) * original.stride_E + col] =
                            Int8NtDequantConvert<BFloat>::run(v[decltype(k)::value],
                                                              factor[decltype(i)::value]);
                    });
                });
            });
        }
        else
        {
            float factor[2][4];
            static_for<0, 2, 1>{}([&](auto j) {
                static_for<0, 4, 1>{}([&](auto k) {
                    const index_t col =
                        nb + consumer * 32 + decltype(j)::value * 16 + h * 4 + decltype(k)::value;
                    factor[decltype(j)::value][decltype(k)::value] =
                        sa * (ChannelB ? scale.b[col] : sb);
                });
            });
            static_for<0, P::BM / 16, 1>{}([&](auto i) {
                const index_t row = mb + decltype(i)::value * 16 + lane % 16;
                static_for<0, 2, 1>{}([&](auto j) {
                    const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                    const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    const bool b0 = h & 1, b1 = h & 2;
                    int32_t t0    = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1),
                            t1    = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
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
                    using V       = uint16_t __attribute__((ext_vector_type(4)));
                    const auto* f = factor[decltype(j)::value];
                    const V out{Int8NtDequantConvert<BFloat>::run(r0, f[0]),
                                Int8NtDequantConvert<BFloat>::run(r1, f[1]),
                                Int8NtDequantConvert<BFloat>::run(r2, f[2]),
                                Int8NtDequantConvert<BFloat>::run(r3, f[3])};
                    *reinterpret_cast<V*>(static_cast<uint16_t*>(original.e_ptr) +
                                          static_cast<long long>(row) * original.stride_E + base +
                                          h * 4) = out;
                });
            });
        }
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& original,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        if(wave < P::ProducerWaves)
            return;
        if(scale.a_stride == 0)
        {
            const float sa = scale.a ? scale.a[0] : 1.0f;
            if(scale.b && scale.b_stride == 1)
                scalar_a<true>(original, mb, nb, tile, scale, sa, 1.0f);
            else
            {
                const float sb = scale.b ? scale.b[0] : 1.0f;
                scalar_a<false>(original, mb, nb, tile, scale, sa, sb);
            }
            return;
        }
        const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
        const Int8NtDequantStore<BFloat> store{scale};
        static_for<0, P::BM / 16, 1>{}([&](auto i) {
            const index_t row = mb + decltype(i)::value * 16 + lane % 16;
            static_for<0, 2, 1>{}([&](auto j) {
                const auto v       = tile.v[decltype(i)::value * 2 + decltype(j)::value];
                const index_t base = nb + consumer * 32 + decltype(j)::value * 16;
                if constexpr(P::Transpose)
                {
                    static_for<0, 4, 1>{}([&](auto k) {
                        const index_t col = base + h + decltype(k)::value * 4;
                        if(col < original.M && row < original.N)
                            store.one(original, col, row, v[decltype(k)::value]);
                    });
                }
                else
                {
                    if(row >= original.M || base + h * 4 + 4 > original.N)
                        return;
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    const bool b0 = h & 1, b1 = h & 2;
                    int32_t t0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1),
                            t1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
                    r0         = b0 ? t0 : r0;
                    r1         = b0 ? r1 : t0;
                    r2         = b0 ? t1 : r2;
                    r3         = b0 ? r3 : t1;
                    t0         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r2);
                    t1         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r1 : r3);
                    r0         = b1 ? t0 : r0;
                    r2         = b1 ? r2 : t0;
                    r1         = b1 ? t1 : r1;
                    r3         = b1 ? r3 : t1;
                    store.four(original, row, base + h * 4, r0, r1, r2, r3);
                }
            });
        });
    }
};
} // namespace ck_tile
