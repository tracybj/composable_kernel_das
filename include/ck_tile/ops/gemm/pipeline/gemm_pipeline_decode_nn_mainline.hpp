// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

// Independent dense NN decode dataflow. A CTA owns every row (M<=64).
// N waves own disjoint 32-column strips, reusing each B panel across all M
// fragments. A is in registers; B uses wave-private LDS. The global ring and
// the optional two-bank LDS operand lookahead are separate scheduling levels.
// No CTA barrier, M split, K split, or partial-output reduction is needed.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_mainline {
using Bf16x4 = uint16_t __attribute__((ext_vector_type(4)));
using Bf16x8 = uint16_t __attribute__((ext_vector_type(8)));
using F32x4  = float __attribute__((ext_vector_type(4)));
using U32x4  = uint32_t __attribute__((ext_vector_type(4)));
using Srd    = int32_t __attribute__((ext_vector_type(4)));

struct Params
{
    const uint16_t* a;
    const uint16_t* b;
    uint16_t* c;
    int m, n, k, lda, ldb, ldc, n0;
};

CK_TILE_DEVICE Srd descriptor(const void* pointer, uint32_t bytes)
{
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return Srd{(int32_t)address, (int32_t)(address >> 32), (int32_t)bytes, 0x00020000};
}

CK_TILE_DEVICE Bf16x4 load_a(Srd srd, uint64_t pair, uint32_t offset)
{
    Bf16x4 value;
    asm volatile("buffer_load_dwordx2 %0, %1, %2, %3 idxen offen\n"
                 : "=v"(value) : "v"(pair), "s"(srd), "s"(offset) : "memory");
    return value;
}

CK_TILE_DEVICE void load_b(Srd srd, uint32_t offset, uint32_t stage, uint32_t base)
{
    asm volatile("s_mov_b32 m0, %3\n\t"
                 "buffer_load_dwordx4 %0, %1, %2 offen lds\n"
                 :: "v"(offset), "s"(srd), "s"(stage), "s"(base) : "memory");
}

// Tie the asynchronously loaded A fragments to the wait. Their consumers
// cannot be scheduled before this definition of the operand registers.
template<int Pending, int Rows>
CK_TILE_DEVICE void wait_global(Bf16x4 (&a)[Rows])
{
    static_assert(Pending >= 0 && Pending <= 63);
    if constexpr(Rows == 1)
        asm volatile("s_waitcnt vmcnt(%1)\n" : "+v"(a[0]) : "n"(Pending) : "memory");
    else if constexpr(Rows == 2)
        asm volatile("s_waitcnt vmcnt(%2)\n" : "+v"(a[0]), "+v"(a[1])
                     : "n"(Pending) : "memory");
    else
        asm volatile("s_waitcnt vmcnt(%4)\n"
                     : "+v"(a[0]), "+v"(a[1]), "+v"(a[2]), "+v"(a[3])
                     : "n"(Pending) : "memory");
}

CK_TILE_DEVICE Bf16x8 read_b(uint32_t address)
{
    Bf16x8 value;
    asm volatile("ds_read_m32x16_b16 %0, %1\n"
                 : "=v"(value) : "v"(address) : "memory");
    return value;
}

template<int Pending>
CK_TILE_DEVICE void wait_operand(Bf16x8& value)
{
    asm volatile("s_waitcnt lgkmcnt(%1)\n" : "+v"(value) : "n"(Pending) : "memory");
}

// Volatile MMAC and tied outputs keep the intended issue/read/wait/compute
// order visible to LLVM, including the ordering between adjacent iterations.
CK_TILE_DEVICE void mmac(F32x4& c, const Bf16x4& a, const Bf16x4& b)
{
    asm volatile("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0\n"
                 : "+v"(c) : "v"(a), "v"(b) : "memory");
}

template<int Rows>
CK_TILE_DEVICE void compute(const Bf16x4 (&a)[Rows], const Bf16x8& b,
                            F32x4 (&c)[Rows][2])
{
    const Bf16x4 b0 = __builtin_shufflevector(b, b, 0, 1, 2, 3);
    const Bf16x4 b1 = __builtin_shufflevector(b, b, 4, 5, 6, 7);
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        mmac(c[m][0], a[m], b0);
        mmac(c[m][1], a[m], b1);
    });
}

template<int Rows, int Slot>
CK_TILE_DEVICE void prefetch(Srd a_srd, const uint64_t (&pairs)[Rows], Srd b_srd,
                             uint32_t b_offset, uint32_t wave_base, int ldb,
                             int k, Bf16x4 (&a)[Rows])
{
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        a[m] = load_a(a_srd, pairs[m], (uint32_t)k * 2u);
    });
    load_b(b_srd, b_offset, (uint32_t)k * (uint32_t)ldb * 2u,
           wave_base + Slot * 1024u);
}

CK_TILE_DEVICE uint16_t to_bf16(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

template<int Rows, int Depth, bool OperandLookahead>
CK_TILE_DEVICE void run(const Params& p)
{
    static_assert(Rows == 1 || Rows == 2 || Rows == 4);
    static_assert(Depth == 2 || Depth == 4 || Depth == 8);
    static_assert(!OperandLookahead || Depth >= 4);
    constexpr int Requests = Rows + 1;
    const uint32_t lane = (uint32_t)threadIdx.x & 63u;
    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    const uint32_t wave_base = (uint32_t)wave * Depth * 1024u;
    auto a_srd = descriptor(p.a, (uint32_t)((int64_t)p.m * p.lda * 2));
    const auto b_srd = descriptor(p.b, (uint32_t)((int64_t)p.k * p.ldb * 2));
    const uint32_t pitch = (uint32_t)p.lda * 2u;
    uint32_t stride = 64u;
    while(stride * 2u <= pitch && stride * 2u <= 0x3fffu) stride *= 2u;
    a_srd[1] |= (int32_t)(stride << 16);
#if defined(__gfx936__) || defined(__gfx938__)
    a_srd[1] |= (int32_t)(1u << 30);
#endif
    uint64_t pairs[Rows];
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        const uint32_t row = m * 16u + (lane & 15u);
        const uint32_t index = row < (uint32_t)p.m ? row : 0xffffffffu;
        const uint32_t offset = row * (pitch - stride) + (lane >> 4) * 8u;
        pairs[m] = (uint64_t)index | ((uint64_t)offset << 32);
    });
    const uint32_t bk = lane >> 2;
    const uint32_t octet = (lane & 3u) ^ ((bk & 1u) * 2u);
    const uint32_t b_offset = (bk * (uint32_t)p.ldb + (uint32_t)p.n0 +
                               (uint32_t)wave * 32u + octet * 8u) * 2u;
    const uint32_t read_offset = bk * 64u + octet * 16u;
    Bf16x4 a[Depth][Rows];
    F32x4 c[Rows][2];
    static_for<0, Rows * 2, 1>{}([&](auto ci) {
        constexpr int i = decltype(ci)::value;
        c[i / 2][i % 2] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        prefetch<Rows, j>(a_srd, pairs, b_srd, b_offset, wave_base, p.ldb, j * 16, a[j]);
    });

    // Host contract requires K%(Depth*16)==0 and K>=Depth*16. Every initial
    // and steady prefetch is necessary; the final window is drained without
    // issuing global requests. All ring indices are compile-time constants.
    if constexpr(OperandLookahead)
    {
        Bf16x8 operands[2];
        wait_global<Requests * (Depth - 1)>(a[0]);
        operands[0] = read_b(wave_base + read_offset);
        for(int k = 0; k + Depth * 16 < p.k; k += Depth * 16)
        {
            static_for<0, Depth, 1>{}([&](auto si) {
                constexpr int j = decltype(si)::value;
                constexpr int next = (j + 1) % Depth;
                wait_global<Requests * (Depth - 2)>(a[j]);
                operands[(j + 1) % 2] = read_b(wave_base + next * 1024u + read_offset);
                wait_operand<1>(operands[j % 2]);
                compute(a[j], operands[j % 2], c);
                prefetch<Rows, j>(a_srd, pairs, b_srd, b_offset, wave_base, p.ldb,
                                   k + (Depth + j) * 16, a[j]);
            });
        }
        static_for<0, Depth, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            if constexpr(j + 1 < Depth)
            {
                wait_global<Requests * (Depth - 2 - j)>(a[j]);
                operands[(j + 1) % 2] = read_b(wave_base + (j + 1) * 1024u + read_offset);
                wait_operand<1>(operands[j % 2]);
            }
            else
            {
                wait_global<0>(a[j]);
                wait_operand<0>(operands[j % 2]);
            }
            compute(a[j], operands[j % 2], c);
        });
    }
    else
    {
        for(int k = 0; k + Depth * 16 < p.k; k += Depth * 16)
        {
            static_for<0, Depth, 1>{}([&](auto si) {
                constexpr int j = decltype(si)::value;
                wait_global<Requests * (Depth - 1)>(a[j]);
                auto b = read_b(wave_base + j * 1024u + read_offset);
                wait_operand<0>(b);
                compute(a[j], b, c);
                prefetch<Rows, j>(a_srd, pairs, b_srd, b_offset, wave_base, p.ldb,
                                   k + (Depth + j) * 16, a[j]);
            });
        }
        static_for<0, Depth, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            wait_global<Requests * (Depth - 1 - j)>(a[j]);
            auto b = read_b(wave_base + j * 1024u + read_offset);
            wait_operand<0>(b);
            compute(a[j], b, c);
        });
    }

    // The retired B ring supplies 1KiB per wave for a BF16 output transpose.
    // Each M fragment is written and read before that same private area is
    // reused. There is no communication or reduction between waves.
    const uint32_t row_lane = lane & 15u;
    const uint32_t q = lane >> 4;
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<0, 8, 1>{}([&](auto ei) {
            constexpr int e = decltype(ei)::value;
            const uint32_t byte = (q + (uint32_t)((e / 4) * 16 + (e % 4) * 4)) * 2u;
            const uint32_t address = wave_base + row_lane * 64u +
                (((byte >> 4) ^ (row_lane & 3u)) << 4) + (byte & 15u);
            const uint32_t value = to_bf16(c[m][e / 4][e % 4]);
            asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
        });
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const uint32_t row = lane >> 2;
        const uint32_t unit = lane & 3u;
        const uint32_t address = wave_base + row * 64u + ((unit ^ (row & 3u)) << 4);
        U32x4 packed;
        asm volatile("ds_read_b128 %0, %1\n" : "=v"(packed) : "v"(address) : "memory");
        asm volatile("s_waitcnt lgkmcnt(0)\n" : "+v"(packed) :: "memory");
        if(m * 16u + row < (uint32_t)p.m)
            *reinterpret_cast<U32x4*>(p.c + (int64_t)(m * 16u + row) * p.ldc +
                                      p.n0 + wave * 32 + unit * 8u) = packed;
    });
}
} // namespace decode_nn_mainline
} // namespace ck_tile
#pragma clang diagnostic pop
