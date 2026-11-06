// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include "ck_tile/core.hpp"

// Independent decode dataflow: one wave owns the entire16/32/64x128 tile.
// A stays in VGPR; B uses private4KiB LDS panels. A is reused across all N
// fragments, B across all M fragments, and global B loads cover256B rows.
// There is no inter-wave communication, K reduction or CTA barrier.
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
CK_TILE_DEVICE U32x4 read_packed(uint32_t address)
{
    U32x4 value;
    asm volatile("ds_read_b128 %0, %1\n" : "=v"(value) : "v"(address) : "memory");
    return value;
}
CK_TILE_DEVICE uint16_t to_bf16(float value)
{
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}
template<int Rows>
CK_TILE_DEVICE void consume(const Bf16x4 (&a)[Rows], uint32_t slot,
                            const uint32_t (&addresses)[4], F32x4 (&c)[Rows][8])
{
    Bf16x8 b[4];
    static_for<0, 4, 1>{}([&](auto index) {
        constexpr int n = decltype(index)::value;
        b[n] = read_b(slot + addresses[n]);
    });
    // The dependency must be visible to LLVM: a memory clobber alone does
    // not prevent register-only MMAC intrinsics moving above this wait.
    asm volatile("s_waitcnt lgkmcnt(0)\n"
                 : "+v"(b[0]), "+v"(b[1]), "+v"(b[2]), "+v"(b[3]) :: "memory");
    static_for<0, Rows * 8, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value / 8;
        constexpr int n = decltype(index)::value % 8;
        Bf16x4 operand;
        if constexpr(n % 2 == 0)
            operand = __builtin_shufflevector(b[n / 2], b[n / 2], 0, 1, 2, 3);
        else
            operand = __builtin_shufflevector(b[n / 2], b[n / 2], 4, 5, 6, 7);
        c[m][n] = __builtin_hcu_mmac_f32_16x16x16_bf16(a[m], operand, c[m][n]);
    });
}

template<int Rows, int Slot>
CK_TILE_DEVICE void prefetch(Srd a_srd, const uint64_t (&a_pair)[Rows], Srd b_srd,
                             uint32_t b_offset, int ldb, int k, Bf16x4 (&a)[Rows])
{
    static_for<0, Rows, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value;
        a[m] = load_a(a_srd, a_pair[m], (uint32_t)k * 2u);
    });
    static_for<0, 4, 1>{}([&](auto index) {
        constexpr int packet = decltype(index)::value;
        load_b(b_srd, b_offset, (uint32_t)(k + packet * 4) * (uint32_t)ldb * 2u,
               Slot * 4096u + packet * 1024u);
    });
}

template<int Rows, int Depth>
CK_TILE_DEVICE void run(const Params& p)
{
    static_assert(Rows == 1 || Rows == 2 || Rows == 4);
    static_assert(Depth == 2 || Depth == 4 || Depth == 8);
    constexpr int Loads = Rows + 4;
    static_assert(Loads * (Depth - 1) <= 63);
    const int lane = (int)threadIdx.x & 63;
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
        const int row = p.m0 + m * 16 + (lane & 15);
        const uint32_t record = row < p.m ? (uint32_t)row : 0xffffffffu;
        const uint32_t offset = (uint32_t)row * (pitch - stride) + (uint32_t)(lane >> 4) * 8u;
        a_pair[m] = (uint64_t)record | ((uint64_t)offset << 32);
    });
    // One packet contains4K rows x128N, exactly1024B per wave instruction.
    // Four packets fill one16x128 BF16 panel. The row XOR is inverted below.
    const int bk = lane >> 4;
    const int octet = (lane & 15) ^ ((bk & 1) * 4);
    const uint32_t b_offset = (uint32_t)(((int64_t)bk * p.ldb + p.n0 + octet * 8) * 2);
    const uint32_t rk = (uint32_t)(lane >> 2);
    uint32_t read_offsets[4];
    static_for<0, 4, 1>{}([&](auto index) {
        constexpr int n = decltype(index)::value;
        const uint32_t col = (uint32_t)(n * 4 + (lane & 3));
        read_offsets[n] = rk * 256u + ((col ^ (4u * (rk & 1u))) << 4);
    });
    Bf16x4 a[Depth][Rows];
    F32x4 c[Rows][8];
    static_for<0, Rows * 8, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value / 8;
        constexpr int n = decltype(index)::value % 8;
        c[m][n] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        prefetch<Rows, j>(a_srd, a_pair, b_srd, b_offset, p.ldb, j * 16, a[j]);
    });
    // No bounds branch in a steady phase. The host chooses Depth8 only when
    // K is a multiple128; M64 uses Depth4 for K%64 to fit registers,
    // and Depth2 covers the remaining K%32 contract. A stage
    // issues Rows+4 VMEM requests. It is consumed before its slot is reused.
    for(int k = 0; k + Depth * 16 < p.k; k += Depth * 16)
    {
        static_for<0, Depth, 1>{}([&](auto index) {
            constexpr int j = decltype(index)::value;
            wait_global<Loads * (Depth - 1)>();
            consume(a[j], j * 4096u, read_offsets, c);
            prefetch<Rows, j>(a_srd, a_pair, b_srd, b_offset, p.ldb,
                             k + (Depth + j) * 16, a[j]);
        });
    }
    static_for<0, Depth, 1>{}([&](auto index) {
        constexpr int j = decltype(index)::value;
        wait_global<Loads * (Depth - 1 - j)>();
        consume(a[j], j * 4096u, read_offsets, c);
    });
    // Every global request and DS read is retired. Reuse the first4KiB B
    // panel for one16-row output slice at a time, entirely within one wave.
    const uint32_t row_lane = (uint32_t)(lane & 15);
    const uint32_t q = (uint32_t)(lane >> 4);
    static_for<0, Rows, 1>{}([&](auto index) {
        constexpr int m = decltype(index)::value;
        static_for<0, 8 * 4, 1>{}([&](auto element) {
            constexpr int n = decltype(element)::value / 4;
            constexpr int j = decltype(element)::value % 4;
            const uint32_t byte = (q + (uint32_t)(n * 16 + j * 4)) * 2u;
            const uint32_t address = row_lane * 256u +
                (((byte >> 4) ^ (row_lane & 7u)) << 4) + (byte & 15u);
            const uint32_t value = to_bf16(c[m][n][j]);
            asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
        });
        wait_lds();
        U32x4 packed[4];
        static_for<0, 4, 1>{}([&](auto packet) {
            constexpr int j = decltype(packet)::value;
            const uint32_t t = (uint32_t)lane + 64u * j;
            const uint32_t row = t >> 4;
            const uint32_t unit = t & 15u;
            const uint32_t address = row * 256u + ((unit ^ (row & 7u)) << 4);
            packed[j] = read_packed(address);
        });
        wait_lds();
        static_for<0, 4, 1>{}([&](auto packet) {
            constexpr int j = decltype(packet)::value;
            const uint32_t t = (uint32_t)lane + 64u * j;
            const int row = p.m0 + m * 16 + (int)(t >> 4);
            const uint32_t col = (uint32_t)p.n0 + (t & 15u) * 8u;
            if(row < p.m)
                *reinterpret_cast<U32x4*>(p.c + (int64_t)row * p.ldc + col) = packed[j];
        });
    });
}
} // namespace decode_nn_wave
} // namespace ck_tile
#pragma clang diagnostic pop
