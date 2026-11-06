// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "ck_tile/core.hpp"

// Independent A-LDS/B-register decode prototype. Four waves own disjoint
// 16/32/64x32 output tiles, each traversing the whole K. A K32 panels use
// wave-private LDS; B goes directly from row-major global memory to VGPR.
// No cross-wave exchange, CTA barrier or intermediate C reduction.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_wave {
using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
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
    return Srd{(int32_t)address, (int32_t)(address >> 32), (int32_t)bytes, 0x00020000};
}
CK_TILE_DEVICE void load_a(Srd srd, uint64_t pair, uint32_t offset, uint32_t base)
{
    asm volatile("s_mov_b32 m0, %3\n\t"
                 "buffer_load_dwordx4 %0, %1, %2 idxen offen lds\n"
                 :: "v"(pair), "s"(srd), "s"(offset), "s"(base) : "memory");
}
CK_TILE_DEVICE uint32_t load_b(Srd srd, uint32_t offset, uint32_t stage)
{
    // LLVM must know this is asynchronous VMEM, including any register copies
    // introduced after instruction scheduling. An opaque asm load is unsafe.
    return (uint16_t)llvm_amdgcn_raw_buffer_load_i16(
        bit_cast<int32x4_t>(srd), (index_t)offset, (index_t)stage, 0);
}
template<int Pending>
CK_TILE_DEVICE void wait_global(U32x4 (&b)[2][2])
{
    // Tie the raw32-bit values to the wait, before narrowing/packing them.
    asm volatile("s_waitcnt vmcnt(%4)\n"
                 : "+v"(b[0][0]), "+v"(b[0][1]), "+v"(b[1][0]), "+v"(b[1][1])
                 : "n"(Pending) : "memory");
}
CK_TILE_DEVICE Bf16x4 read_a(uint32_t address)
{
    Bf16x4 value;
    asm volatile("ds_read_b64 %0, %1\n" : "=v"(value) : "v"(address) : "memory");
    asm volatile("s_waitcnt lgkmcnt(0)\n" : "+v"(value) :: "memory");
    return value;
}
CK_TILE_DEVICE uint16_t to_bf16(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
template<int Rows, int Slot>
CK_TILE_DEVICE void prefetch(Srd a_srd, const uint64_t (&a_pair)[Rows], Srd b_srd,
                             uint32_t b_offset, uint32_t wave_base, int ldb,
                             int k, U32x4 (&b)[2][2])
{
    static_for<0, Rows, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value;
        load_a(a_srd, a_pair[m], (uint32_t)k * 2u,
               wave_base + (Slot * Rows + m) * 1024u);
    });
    static_for<0, 4, 1>{}([&](auto index) {
        constexpr int sub = decltype(index)::value / 2;
        constexpr int n = decltype(index)::value % 2;
        const uint32_t base = (uint32_t)(k + sub * 16) * (uint32_t)ldb * 2u + n * 32u;
        const uint32_t q0 = load_b(b_srd, b_offset, base);
        const uint32_t q1 = load_b(b_srd, b_offset, base + (uint32_t)ldb * 2u);
        const uint32_t q2 = load_b(b_srd, b_offset, base + (uint32_t)ldb * 4u);
        const uint32_t q3 = load_b(b_srd, b_offset, base + (uint32_t)ldb * 6u);
        b[sub][n] = U32x4{q0, q1, q2, q3};
    });
}
template<int Rows>
CK_TILE_DEVICE void consume(const U32x4 (&b)[2][2], uint32_t slot,
                            uint32_t lane, F32x4 (&c)[Rows][2])
{
    static_for<0, 2, 1>{}([&](auto step) {
        constexpr int sub = decltype(step)::value;
        Bf16x4 bv[2];
        static_for<0, 2, 1>{}([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            bv[n] = __builtin_convertvector(b[sub][n], Bf16x4);
        });
        static_for<0, Rows, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const uint32_t row = lane & 15u;
            const uint32_t byte = ((lane >> 4) * 4u + sub * 16u) * 2u;
            const uint32_t address = slot + m * 1024u + row * 64u +
                (((byte >> 4) ^ (row & 3u)) << 4) + (byte & 15u);
            const auto av = read_a(address);
            c[m][0] = __builtin_hcu_mmac_f32_16x16x16_bf16(av, bv[0], c[m][0]);
            c[m][1] = __builtin_hcu_mmac_f32_16x16x16_bf16(av, bv[1], c[m][1]);
        });
    });
}
template<int Rows, int Depth>
CK_TILE_DEVICE void run(const Params& p)
{
    static_assert(Rows == 1 || Rows == 2 || Rows == 4);
    static_assert(Depth == 2);
    constexpr int Loads = Rows + 16;
    const uint32_t lane = (uint32_t)threadIdx.x & 63u;
    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    const uint32_t wave_base = (uint32_t)wave * (Depth * Rows * 1024u);
    auto a_srd = descriptor(p.a, (uint32_t)((int64_t)p.m * p.lda * 2));
    const auto b_srd = descriptor(p.b, (uint32_t)((int64_t)p.k * p.ldb * 2));
    const uint32_t pitch = (uint32_t)p.lda * 2u;
    uint32_t stride = 64u;
    while(stride * 2u <= pitch && stride * 2u <= 0x3fffu) stride *= 2u;
    a_srd[1] |= (int32_t)(stride << 16);
#if defined(__gfx936__) || defined(__gfx938__)
    a_srd[1] |= (int32_t)(1u << 30);
#endif
    uint64_t a_pair[Rows];
    static_for<0, Rows, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value;
        const uint32_t row = (uint32_t)p.m0 + m * 16u + (lane >> 2);
        const uint32_t record = row < (uint32_t)p.m ? row : 0xffffffffu;
        const uint32_t octet = (lane & 3u) ^ (row & 3u);
        const uint32_t offset = row * (pitch - stride) + octet * 16u;
        a_pair[m] = (uint64_t)record | ((uint64_t)offset << 32);
    });
    const uint32_t b_offset = ((lane >> 4) * 4u * (uint32_t)p.ldb +
        (uint32_t)p.n0 + (uint32_t)wave * 32u + (lane & 15u)) * 2u;
    U32x4 b[Depth][2][2];
    F32x4 c[Rows][2];
    static_for<0, Rows * 2, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value / 2;
        constexpr int n = decltype(index)::value % 2;
        c[m][n] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        if(j * 32 < p.k)
            prefetch<Rows, j>(a_srd, a_pair, b_srd, b_offset, wave_base, p.ldb, j * 32, b[j]);
    });
    // K32 can have one stage. It is a real cold path, never an unused
    // prefetch beyond the supported K. The common path has two-stage overlap.
    const int stages = p.k / 32;
    int stage = 0;
    for(; stage + Depth < stages; stage += Depth)
    {
        static_for<0, Depth, 1>{}([&](auto index) {
            constexpr int j = decltype(index)::value;
            wait_global<Loads * (Depth - 1)>(b[j]);
            consume(b[j], wave_base + j * Rows * 1024u, lane, c);
            if(stage + Depth + j < stages)
                prefetch<Rows, j>(a_srd, a_pair, b_srd, b_offset, wave_base, p.ldb,
                                 (stage + Depth + j) * 32, b[j]);
        });
    }
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        if(stage + j < stages)
        {
            // The final group can have one stage: vmcnt0 is deliberately
            // conservative here until the variable-count drain is proven.
            wait_global<0>(b[j]);
            consume(b[j], wave_base + j * Rows * 1024u, lane, c);
        }
    });
    const uint32_t row_lane = lane & 15u;
    const uint32_t q = lane >> 4;
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<0, 8, 1>{}([&](auto element) {
            constexpr int n = decltype(element)::value / 4;
            constexpr int j = decltype(element)::value % 4;
            const uint32_t byte = (q + (uint32_t)(n * 16 + j * 4)) * 2u;
            const uint32_t address = wave_base + row_lane * 64u +
                (((byte >> 4) ^ (row_lane & 3u)) << 4) + (byte & 15u);
            const uint32_t value = to_bf16(c[m][n][j]);
            asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
        });
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const uint32_t row = lane >> 2;
        const uint32_t unit = lane & 3u;
        const uint32_t address = wave_base + row * 64u + ((unit ^ (row & 3u)) << 4);
        U32x4 packed;
        asm volatile("ds_read_b128 %0, %1\n" : "=v"(packed) : "v"(address) : "memory");
        asm volatile("s_waitcnt lgkmcnt(0)\n" : "+v"(packed) :: "memory");
        const int out_row = p.m0 + m * 16 + (int)row;
        if(out_row < p.m)
            *reinterpret_cast<U32x4*>(p.c + (int64_t)out_row * p.ldc +
                                     p.n0 + wave * 32 + unit * 8u) = packed;
    });
}
} // namespace decode_nn_wave
} // namespace ck_tile
#pragma clang diagnostic pop
