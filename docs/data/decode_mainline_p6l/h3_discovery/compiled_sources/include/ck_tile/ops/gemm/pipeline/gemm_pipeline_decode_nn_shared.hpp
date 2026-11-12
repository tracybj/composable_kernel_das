// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "gemm_pipeline_decode_nn_mainline.hpp"

// Independent complete-M decode pipeline. A is loaded cooperatively once per
// CTA; N waves each load their own B panel and reuse it for every M fragment.
// Global-to-LDS rings and two banks of MMAC operands overlap separately.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_shared {
using namespace decode_nn_mainline;

template<int Rows, int BlockK, int Depth>
struct Geometry
{
    static constexpr int Halves = BlockK / 16;
    static constexpr int ALoadRows = 512 / BlockK;
    static constexpr int ALoadWaves = (Rows * 16 + ALoadRows - 1) / ALoadRows;
    static constexpr int ABytes = ALoadWaves * 1024;
    static constexpr int BWaveBytes = BlockK * 64;
    static constexpr int StageBytes = ABytes + 4 * BWaveBytes;
    static constexpr int LdsBytes = Depth * StageBytes;
    static_assert(Rows == 1 || Rows == 2 || Rows == 4);
    static_assert(BlockK == 16 || BlockK == 32);
    static_assert(Depth == 4 || Depth == 8);
    static_assert(LdsBytes <= 65536);
};

template<int Rows, int BlockK>
struct Operands
{
    Bf16x4 a[BlockK / 16][Rows];
    Bf16x8 b[BlockK / 16];
};

CK_TILE_DEVICE void load_a_lds(Srd srd, uint64_t pair, uint32_t offset, uint32_t base)
{
    asm volatile("s_mov_b32 m0, %3\n\t"
                 "buffer_load_dwordx4 %0, %1, %2 idxen offen slc lds\n"
                 :: "v"(pair), "s"(srd), "s"(offset), "s"(base) : "memory");
}

CK_TILE_DEVICE Bf16x4 read_a(uint32_t address)
{
    Bf16x4 value;
    asm volatile("ds_read_b64 %0, %1\n" : "=v"(value) : "v"(address) : "memory");
    return value;
}

template<int Rows, int BlockK, int Depth, int PendingStages>
CK_TILE_DEVICE void ready(int wave)
{
    using G = Geometry<Rows, BlockK, Depth>;
    constexpr int BRequests = BlockK / 16;
    if(wave < G::ALoadWaves)
        asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"((BRequests + 1) * PendingStages) : "memory");
    else
        asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(BRequests * PendingStages) : "memory");
}

template<int Rows, int BlockK>
CK_TILE_DEVICE void readers_retired(Operands<Rows, BlockK>& current)
{
    // Retire current register loads before all waves release this LDS slot.
    // The barrier also publishes the next global-to-LDS panel across waves.
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n"
                 : "+v"(current.a[0][0]), "+v"(current.b[0]) :: "memory");
}

template<int Rows, int BlockK, int Depth, int Slot>
CK_TILE_DEVICE void prefetch(Srd a_srd, uint64_t a_pair, Srd b_srd,
                            uint32_t b_offset, int wave, int ldb, int k)
{
    using G = Geometry<Rows, BlockK, Depth>;
    if(wave < G::ALoadWaves)
        load_a_lds(a_srd, a_pair, (uint32_t)k * 2u,
                   Slot * G::StageBytes + (uint32_t)wave * 1024u);
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        load_b(b_srd, b_offset, (uint32_t)(k + h * 16) * (uint32_t)ldb * 2u,
               Slot * G::StageBytes + G::ABytes + (uint32_t)wave * G::BWaveBytes + h * 1024u);
    });
}

template<int Rows, int BlockK, int Depth, int Slot>
CK_TILE_DEVICE void read(Operands<Rows, BlockK>& operand,
                        uint32_t lane, int wave, uint32_t b_read)
{
    using G = Geometry<Rows, BlockK, Depth>;
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        static_for<0, Rows, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            operand.a[h][m] = read_a(Slot * G::StageBytes +
                (m * 16u + (lane & 15u)) * BlockK * 2u + h * 32u + (lane >> 4) * 8u);
        });
        operand.b[h] = read_b(Slot * G::StageBytes + G::ABytes +
                               (uint32_t)wave * G::BWaveBytes + h * 1024u + b_read);
    });
}

template<int Rows, int BlockK>
CK_TILE_DEVICE void calculate(const Operands<Rows, BlockK>& operand, F32x4 (&c)[Rows][2])
{
    static_for<0, BlockK / 16, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        compute(operand.a[h], operand.b[h], c);
    });
}

template<int Rows>
CK_TILE_DEVICE void store(const Params& p, F32x4 (&c)[Rows][2], uint32_t lane, int wave)
{
    const uint32_t base = (uint32_t)wave * 1024u;
    const uint32_t row_lane = lane & 15u;
    const uint32_t q = lane >> 4;
    static_for<0, Rows, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<0, 8, 1>{}([&](auto ei) {
            constexpr int e = decltype(ei)::value;
            const uint32_t byte = (q + (uint32_t)((e / 4) * 16 + (e % 4) * 4)) * 2u;
            const uint32_t address = base + row_lane * 64u +
                (((byte >> 4) ^ (row_lane & 3u)) << 4) + (byte & 15u);
            const uint32_t value = to_bf16(c[m][e / 4][e % 4]);
            asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
        });
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
        const uint32_t row = lane >> 2;
        const uint32_t unit = lane & 3u;
        const uint32_t address = base + row * 64u + ((unit ^ (row & 3u)) << 4);
        U32x4 packed;
        asm volatile("ds_read_b128 %0, %1\n" : "=v"(packed) : "v"(address) : "memory");
        asm volatile("s_waitcnt lgkmcnt(0)\n" : "+v"(packed) :: "memory");
        if(m * 16u + row < (uint32_t)p.m)
            *reinterpret_cast<U32x4*>(p.c + (int64_t)(m * 16u + row) * p.ldc +
                                      p.n0 + wave * 32 + unit * 8u) = packed;
    });
}

template<int Rows, int BlockK, int Depth>
CK_TILE_DEVICE void run(const Params& p)
{
    using G = Geometry<Rows, BlockK, Depth>;
    const uint32_t lane = (uint32_t)threadIdx.x & 63u;
    const int wave = __builtin_amdgcn_readfirstlane((int)threadIdx.x >> 6);
    auto a_srd = descriptor(p.a, (uint32_t)((int64_t)p.m * p.lda * 2));
    const auto b_srd = descriptor(p.b, (uint32_t)((int64_t)p.k * p.ldb * 2));
    const uint32_t pitch = (uint32_t)p.lda * 2u;
    uint32_t stride = 64u;
    while(stride * 2u <= pitch && stride * 2u <= 0x3fffu) stride *= 2u;
    a_srd[1] |= (int32_t)(stride << 16);
#if defined(__gfx936__) || defined(__gfx938__)
    a_srd[1] |= (int32_t)(1u << 30);
#endif
    constexpr int Chunks = BlockK / 8;
    const uint32_t a_row = (uint32_t)wave * G::ALoadRows + lane / Chunks;
    const uint32_t a_index = a_row < (uint32_t)p.m ? a_row : 0xffffffffu;
    const uint32_t a_offset = a_row * (pitch - stride) + (lane % Chunks) * 16u;
    const uint64_t a_pair = (uint64_t)a_index | ((uint64_t)a_offset << 32);
    const uint32_t bk = lane >> 2;
    const uint32_t octet = (lane & 3u) ^ ((bk & 1u) * 2u);
    const uint32_t b_offset = (bk * (uint32_t)p.ldb + (uint32_t)p.n0 +
                               (uint32_t)wave * 32u + octet * 8u) * 2u;
    const uint32_t b_read = bk * 64u + octet * 16u;
    F32x4 c[Rows][2];
    static_for<0, Rows * 2, 1>{}([&](auto ci) {
        constexpr int i = decltype(ci)::value;
        c[i / 2][i % 2] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        prefetch<Rows, BlockK, Depth, j>(a_srd, a_pair, b_srd, b_offset, wave, p.ldb, j * BlockK);
    });
    ready<Rows, BlockK, Depth, Depth - 1>(wave);
    asm volatile("s_barrier\n" ::: "memory");
    Operands<Rows, BlockK> operands[2];
    read<Rows, BlockK, Depth, 0>(operands[0], lane, wave, b_read);
    for(int k = 0; k + Depth * BlockK < p.k; k += Depth * BlockK)
    {
        static_for<0, Depth, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            constexpr int next = (j + 1) % Depth;
            ready<Rows, BlockK, Depth, Depth - 2>(wave);
            readers_retired(operands[j % 2]);
            read<Rows, BlockK, Depth, next>(operands[(j + 1) % 2], lane, wave, b_read);
            prefetch<Rows, BlockK, Depth, j>(a_srd, a_pair, b_srd, b_offset, wave,
                                             p.ldb, k + (Depth + j) * BlockK);
            calculate(operands[j % 2], c);
        });
    }
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        if constexpr(j + 1 < Depth)
        {
            ready<Rows, BlockK, Depth, Depth - 2 - j>(wave);
            readers_retired(operands[j % 2]);
            read<Rows, BlockK, Depth, j + 1>(operands[(j + 1) % 2], lane, wave, b_read);
        }
        else
        {
            ready<Rows, BlockK, Depth, 0>(wave);
            readers_retired(operands[j % 2]);
        }
        calculate(operands[j % 2], c);
    });
    // All global requests and LDS reads have retired before output reuses LDS.
    store(p, c, lane, wave);
}
} // namespace decode_nn_shared
} // namespace ck_tile
#pragma clang diagnostic pop
