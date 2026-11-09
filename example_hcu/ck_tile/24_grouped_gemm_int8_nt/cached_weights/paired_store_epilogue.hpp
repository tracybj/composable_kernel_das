// SPDX-License-Identifier: MIT
#pragma once

#include "scaled_epilogue.hpp"

namespace ck_tile {
namespace int8_nt_paired_store {
using FloatPair = ext_vector_t<float, 2>;

CK_TILE_DEVICE FloatPair multiply(FloatPair values, FloatPair factors)
{
    FloatPair result;
    asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(result) : "v"(values), "v"(factors));
    return result;
}

template<bool BFloat>
CK_TILE_DEVICE uint16_t convert(float result)
{
    if constexpr(BFloat)
    {
        const uint32_t bits = bit_cast<uint32_t>(result);
        const uint32_t rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
        const uint32_t nan = (bits >> 16) | 0x40U;
        uint32_t selected;
        asm volatile("v_cmp_u_f32 vcc, %1, %1\n\t"
                     "v_cndmask_b32 %0, %2, %3, vcc"
                     : "=v"(selected)
                     : "v"(result), "v"(rounded), "v"(nan)
                     : "vcc");
        return static_cast<uint16_t>(selected);
    }
    else
        return bit_cast<uint16_t>(type_convert<half_t>(result));
}

template<bool BFloat>
CK_TILE_DEVICE uint32_t scale_pair(int32_t first, int32_t second, FloatPair factors)
{
    const FloatPair values{static_cast<float>(first), static_cast<float>(second)};
    const FloatPair result = multiply(values, factors);
    return uint32_t(convert<BFloat>(result[0])) |
           (uint32_t(convert<BFloat>(result[1])) << 16);
}

// Swap one lane bit and one register index bit. Both halves of each packed
// pair travel together. Every lane participates before the store.
template<int Mask>
CK_TILE_DEVICE void exchange(int lane, uint32_t& lower, uint32_t& upper)
{
    const bool high_lane = (lane & Mask) != 0;
    const auto other = __builtin_amdgcn_ds_bpermute((lane ^ Mask) * 4,
                                                    high_lane ? lower : upper);
    lower = high_lane ? other : lower;
    upper = high_lane ? upper : other;
}
} // namespace int8_nt_paired_store

template<class Problem, bool BFloat>
struct GroupedInt8NtPairedStoreEpilogue
{
    using Tile = typename Problem::Pipeline::AccTile;

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   const Tile& tile,
                                   Int8NtScaleArgs scale) const
    {
        using namespace int8_nt_paired_store;
        static_assert(!Problem::Transpose);
        if(scale.a_stride != 0 || arg.stride_E % 8 != 0 ||
           reinterpret_cast<uintptr_t>(arg.e_ptr) % 16 != 0)
        {
            GroupedInt8NtScaledWaspEpilogue<Problem, BFloat>{}(arg, mb, nb, tile, scale);
            return;
        }
        const int lane = get_thread_id() % 64;
        const int consumer = get_thread_id() / 64 - Problem::ProducerWaves;
        const int stripe = nb + consumer * 32;
        if(consumer < 0 || stripe >= arg.N)
            return;

        // Factors use original MMAC columns. Preserve the FP32 multiplication
        // boundary before applying them to the integer accumulators.
        const float activation_scale = scale.a ? scale.a[0] : 1.0f;
        FloatPair factors[4];
        static_for<0, 4, 1>{}([&](auto pair) {
            constexpr int p = decltype(pair)::value;
            const int col = stripe + lane / 16 + p * 8;
            const float left = scale.b ? scale.b[col * scale.b_stride] : 1.0f;
            const float right = scale.b ? scale.b[(col + 4) * scale.b_stride] : 1.0f;
            factors[p] = multiply(FloatPair{activation_scale, activation_scale},
                                  FloatPair{left, right});
        });
        static_for<0, Problem::BM / 16, 1>{}([&](auto block) {
            constexpr int i = decltype(block)::value;
            const auto first = tile.v[i * 2];
            const auto second = tile.v[i * 2 + 1];
            uint32_t p0 = scale_pair<BFloat>(first[0], first[1], factors[0]);
            uint32_t p1 = scale_pair<BFloat>(first[2], first[3], factors[1]);
            uint32_t p2 = scale_pair<BFloat>(second[0], second[1], factors[2]);
            uint32_t p3 = scale_pair<BFloat>(second[2], second[3], factors[3]);
            exchange<16>(lane, p0, p1);
            exchange<16>(lane, p2, p3);
            exchange<32>(lane, p0, p2);
            exchange<32>(lane, p1, p3);

            using Output = ext_vector_t<uint16_t, 8>;
            const Output out{uint16_t(p0), uint16_t(p1), uint16_t(p2), uint16_t(p3),
                             uint16_t(p0 >> 16), uint16_t(p1 >> 16),
                             uint16_t(p2 >> 16), uint16_t(p3 >> 16)};
            const int row = mb + i * 16 + lane % 16;
            const int col = stripe + (lane / 16) * 8;
            auto* dst = static_cast<uint16_t*>(arg.e_ptr) +
                        static_cast<long long>(row) * arg.stride_E + col;
            *reinterpret_cast<Output*>(dst) = out;
        });
    }
};
} // namespace ck_tile
