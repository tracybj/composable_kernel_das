// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "gemm_pipeline_decode_nn_shared.hpp"

// Controlled refinements of the shared-A / wave-private-B dataflow. The old
// pipeline remains instantiated unchanged when both experimental bits are off.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_shared_tuned {
using namespace decode_nn_mainline;
using decode_nn_shared::Geometry;
using decode_nn_shared::Operands;

template<int Rows, int BlockK, int Depth, int Slot>
CK_TILE_DEVICE void prefetch_a(Srd a, uint64_t pair, int wave, int k)
{
    using G = Geometry<Rows, BlockK, Depth>;
    if(wave < G::ALoadWaves)
        decode_nn_shared::load_a_lds(a, pair, (uint32_t)k * 2u,
                                     Slot * G::StageBytes + (uint32_t)wave * 1024u);
}

template<int Rows, int BlockK, int Depth, int Slot>
CK_TILE_DEVICE void prefetch_b(Srd b, uint32_t offset, int wave, int ldb, int k)
{
    using G = Geometry<Rows, BlockK, Depth>;
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        load_b(b, offset, (uint32_t)(k + h * 16) * (uint32_t)ldb * 2u,
               Slot * G::StageBytes + G::ABytes + (uint32_t)wave * G::BWaveBytes + h * 1024u);
    });
}

template<int Rows, int BlockK, int Depth, int Slot, bool XorA>
CK_TILE_DEVICE void read(Operands<Rows, BlockK>& operand,
                         uint32_t lane, int wave, uint32_t b_read)
{
    using G = Geometry<Rows, BlockK, Depth>;
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        static_for<0, Rows, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const uint32_t row = m * 16u + (lane & 15u);
            uint32_t offset = h * 32u + (lane >> 4) * 8u;
            if constexpr(XorA) offset ^= ((row >> 2) & (BlockK / 8u - 1u)) * 16u;
            operand.a[h][m] = decode_nn_shared::read_a(
                Slot * G::StageBytes + row * BlockK * 2u + offset);
        });
        operand.b[h] = read_b(Slot * G::StageBytes + G::ABytes +
            (uint32_t)wave * G::BWaveBytes + h * 1024u + b_read);
    });
}

template<int Rows, int BlockK>
CK_TILE_DEVICE void retire_private_readers(Operands<Rows, BlockK>& current)
{
    asm volatile("s_waitcnt lgkmcnt(0)\n"
                 : "+v"(current.a[0][0]), "+v"(current.b[0]) :: "memory");
}

template<int Rows, int BlockK, int Depth>
CK_TILE_DEVICE void ready_after_early_b(int wave)
{
    using G = Geometry<Rows, BlockK, Depth>;
    constexpr int B = BlockK / 16;
    // The freshly issued B requests are after the next panel in VMEM order.
    if(wave < G::ALoadWaves)
        asm volatile("s_waitcnt vmcnt(%0)\n"
                     :: "n"((B + 1) * (Depth - 2) + B) : "memory");
    else
        asm volatile("s_waitcnt vmcnt(%0)\n"
                     :: "n"(B * (Depth - 2) + B) : "memory");
}

template<int Rows, int BlockK, int Depth, bool EarlyB, bool XorA>
CK_TILE_DEVICE void run(const Params& p)
{
    using G = Geometry<Rows, BlockK, Depth>;
    static_assert(EarlyB || XorA, "The unchanged path uses its original kernel");
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
    uint32_t chunk = lane % Chunks;
    if constexpr(XorA) chunk ^= (a_row >> 2) & (Chunks - 1u);
    const uint32_t a_offset = a_row * (pitch - stride) + chunk * 16u;
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
        prefetch_a<Rows, BlockK, Depth, j>(a_srd, a_pair, wave, j * BlockK);
        prefetch_b<Rows, BlockK, Depth, j>(b_srd, b_offset, wave, p.ldb, j * BlockK);
    });
    decode_nn_shared::ready<Rows, BlockK, Depth, Depth - 1>(wave);
    asm volatile("s_barrier\n" ::: "memory");
    Operands<Rows, BlockK> operands[2];
    read<Rows, BlockK, Depth, 0, XorA>(operands[0], lane, wave, b_read);
    for(int k = 0; k + Depth * BlockK < p.k; k += Depth * BlockK)
    {
        static_for<0, Depth, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            constexpr int next = (j + 1) % Depth;
            const int future_k = k + (Depth + j) * BlockK;
            if constexpr(EarlyB)
            {
                // Only this wave reads its B slot. Retire its current readers,
                // then overlap B refill with the next shared-A arrival wait.
                retire_private_readers(operands[j % 2]);
                prefetch_b<Rows, BlockK, Depth, j>(b_srd, b_offset, wave, p.ldb, future_k);
                ready_after_early_b<Rows, BlockK, Depth>(wave);
                asm volatile("s_barrier\n" ::: "memory");
            }
            else
            {
                decode_nn_shared::ready<Rows, BlockK, Depth, Depth - 2>(wave);
                decode_nn_shared::readers_retired(operands[j % 2]);
            }
            read<Rows, BlockK, Depth, next, XorA>(operands[(j + 1) % 2], lane, wave, b_read);
            prefetch_a<Rows, BlockK, Depth, j>(a_srd, a_pair, wave, future_k);
            if constexpr(!EarlyB)
                prefetch_b<Rows, BlockK, Depth, j>(b_srd, b_offset, wave, p.ldb, future_k);
            decode_nn_shared::calculate(operands[j % 2], c);
        });
    }
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        if constexpr(j + 1 < Depth)
        {
            decode_nn_shared::ready<Rows, BlockK, Depth, Depth - 2 - j>(wave);
            decode_nn_shared::readers_retired(operands[j % 2]);
            read<Rows, BlockK, Depth, j + 1, XorA>(operands[(j + 1) % 2], lane, wave, b_read);
        }
        else
        {
            decode_nn_shared::ready<Rows, BlockK, Depth, 0>(wave);
            decode_nn_shared::readers_retired(operands[j % 2]);
        }
        decode_nn_shared::calculate(operands[j % 2], c);
    });
    decode_nn_shared::store(p, c, lane, wave);
}
} // namespace decode_nn_shared_tuned
} // namespace ck_tile
#pragma clang diagnostic pop
