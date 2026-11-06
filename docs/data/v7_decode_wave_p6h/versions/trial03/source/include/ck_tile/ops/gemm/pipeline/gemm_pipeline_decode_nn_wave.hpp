// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

// Independent decode dataflow. Four waves own disjoint 16x32 output tiles,
// each traversing the complete K axis. A bypasses LDS. Each wave exclusively
// owns its compile-time-depth 1KiB B buffers; neither the main loop nor the epilogue exchanges
// data with another wave. No K split, CTA barrier, or FP32 partial reduction.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_wave {
using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
using Bf16x8 = uint16_t __attribute__((ext_vector_type(8)));
using F32x4 = float __attribute__((ext_vector_type(4)));
using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
using Srd = int32_t __attribute__((ext_vector_type(4)));

struct Params
{
    const uint16_t* a;
    const uint16_t* b;
    uint16_t* c;
    int m, n, k, lda, ldb, ldc, m0, n0;
};

CK_TILE_DEVICE Srd descriptor(const void* ptr, uint32_t bytes)
{
    const auto address = reinterpret_cast<uint64_t>(ptr);
    // Raw byte addressing, no structured stride or cache-swizzle semantics.
    return Srd{(int32_t)address, (int32_t)(address >> 32), (int32_t)bytes, 0x00020000};
}

CK_TILE_DEVICE Bf16x4 load_a(Srd srd, uint64_t pair, uint32_t stage_offset)
{
    Bf16x4 value;
    asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 idxen offen\n"
                 : "=v"(value) : "v"(pair), "s"(srd), "s"(stage_offset) : "memory");
    return value;
}

CK_TILE_DEVICE void load_b(Srd srd, uint32_t offset, uint32_t stage_offset,
                           uint32_t lds_base)
{
    asm volatile("s_mov_b32 m0, %3\n\t"
                 "buffer_load_dwordx4 %0, %1, %2 offen lds\n"
                 :: "v"(offset), "s"(srd), "s"(stage_offset), "s"(lds_base)
                 : "memory");
}

template<int Pending>
CK_TILE_DEVICE void wait_global()
{
    asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(Pending) : "memory");
}
CK_TILE_DEVICE void wait_lds()
{
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
}
CK_TILE_DEVICE Bf16x8 read_b(uint32_t address)
{
    Bf16x8 value;
    asm volatile("ds_read_m32x16_b16 %0, %1\n" : "=v"(value) : "v"(address) : "memory");
    return value;
}
CK_TILE_DEVICE uint16_t to_bf16(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
CK_TILE_DEVICE void consume(const Bf16x4& a, uint32_t address, F32x4& c0, F32x4& c1)
{
    const auto b = read_b(address);
    wait_lds();
    c0 = __builtin_hcu_mmac_f32_16x16x16_bf16(
        a, __builtin_shufflevector(b, b, 0, 1, 2, 3), c0);
    c1 = __builtin_hcu_mmac_f32_16x16x16_bf16(
        a, __builtin_shufflevector(b, b, 4, 5, 6, 7), c1);
}

template<int Depth>
CK_TILE_DEVICE void run(const Params& p)
{
    static_assert(Depth == 2 || Depth == 8);
    const int lane = (int)threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    const uint32_t wave_base = (uint32_t)wave * (Depth * 1024u);
    auto a_srd = descriptor(p.a, (uint32_t)((int64_t)p.m * p.lda * 2));
    const auto b_srd = descriptor(p.b, (uint32_t)((int64_t)p.k * p.ldb * 2));
    const int row = p.m0 + (lane & 15);
    // Structured addressing retains the same physical bytes while letting
    // CACHE_SWIZZLE distribute K-strided A rows across cache tag sets. Raw
    // per-lane row*pitch loads produced new tag-conflict stalls at K2048.
    const uint32_t a_pitch = (uint32_t)p.lda * 2u;
    uint32_t a_stride = 64u;
    while(a_stride * 2u <= a_pitch && a_stride * 2u <= 0x3fffu) a_stride *= 2u;
    a_srd[1] |= (int32_t)(a_stride << 16);
#if defined(__gfx936__) || defined(__gfx938__)
    a_srd[1] |= (int32_t)(1u << 30);
#endif
    const uint32_t a_index = row < p.m ? (uint32_t)row : 0xffffffffu;
    const uint32_t a_offset = (uint32_t)row * (a_pitch - a_stride) + (uint32_t)(lane >> 4) * 8u;
    const uint64_t a_pair = (uint64_t)a_index | ((uint64_t)a_offset << 32);
    const int bk = lane >> 2;
    // Matching XOR on the global load and LDS-read addresses preserves the
    // logical 16x32 panel while distributing adjacent rows across bank groups.
    const int octet = (lane & 3) ^ ((bk & 1) * 2);
    const uint32_t b_offset = (uint32_t)(((int64_t)bk * p.ldb + p.n0 + wave * 32 + octet * 8) * 2);
    const uint32_t read_offset = (uint32_t)(bk * 64 + octet * 16);
    Bf16x4 a[Depth];
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        a[j] = load_a(a_srd, a_pair, j * 32u);
        load_b(b_srd, b_offset, j * 32u * (uint32_t)p.ldb,
               wave_base + j * 1024u);
    });
    F32x4 c0{0, 0, 0, 0}, c1{0, 0, 0, 0};
    // The host selects Depth8 only for K>=128 and K%128==0, otherwise
    // Depth2 for K>=32 and K%32==0. Every prefetch is required, including
    // the minimum-stage case. All operand indices below are compile-time.
    for(int k = 0; k + Depth * 16 < p.k; k += Depth * 16)
    {
        static_for<0, Depth, 1>{}([&](auto index) {
            constexpr int j = decltype(index)::value;
            wait_global<2 * (Depth - 1)>();
            consume(a[j], wave_base + j * 1024u + read_offset, c0, c1);
            a[j] = load_a(a_srd, a_pair, (uint32_t)(k + (Depth + j) * 16) * 2u);
            load_b(b_srd, b_offset,
                   (uint32_t)(k + (Depth + j) * 16) * (uint32_t)p.ldb * 2u,
                   wave_base + j * 1024u);
        });
    }
    // Drain contains no global requests. Waiting for the current wave's DS
    // reads before reusing a slot is sufficient: no other wave owns that slot.
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        wait_global<2 * (Depth - 1 - j)>();
        consume(a[j], wave_base + j * 1024u + read_offset, c0, c1);
    });

    // Wave-private BF16 output transpose. The B ring is fully retired, and no
    // other wave can touch this private region. Only 1KiB is needed for the store.
    const uint32_t r = (uint32_t)(lane & 15);
    const uint32_t q = (uint32_t)(lane >> 4);
#pragma unroll
    for(int n = 0; n < 2; ++n)
    {
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            const uint32_t byte = (q + (uint32_t)(n * 16 + j * 4)) * 2u;
            const uint32_t address = wave_base + r * 64u +
                (((byte >> 4) ^ (r & 3u)) << 4) + (byte & 15u);
            const uint32_t value = to_bf16(n == 0 ? c0[j] : c1[j]);
            asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
        }
    }
    wait_lds();
    const uint32_t out_row = (uint32_t)lane >> 2;
    const uint32_t unit = (uint32_t)lane & 3u;
    const uint32_t address = wave_base + out_row * 64u + ((unit ^ (out_row & 3u)) << 4);
    U32x4 packed;
    asm volatile("ds_read_b128 %0, %1\n" : "=v"(packed) : "v"(address) : "memory");
    wait_lds();
    if(p.m0 + (int)out_row < p.m)
        *reinterpret_cast<U32x4*>(p.c + (int64_t)(p.m0 + (int)out_row) * p.ldc +
                                 p.n0 + wave * 32 + unit * 8u) = packed;
}
} // namespace decode_nn_wave
} // namespace ck_tile
#pragma clang diagnostic pop
