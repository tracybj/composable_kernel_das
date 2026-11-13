// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
// Experimental H3 phase-order factor: compute current MMAC before ready.
// Resource buffers and every VM/LG wait, producer/read and barrier order remain unchanged.
#include "gemm_pipeline_decode_nn_cooperative_compact.hpp"
#include "gemm_pipeline_decode_nn_cooperative_wide.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_resource_phase {

namespace compact_detail {
using namespace decode_nn_cooperative_compact;
template<int Rows, bool ComputeFirst>
CK_TILE_DEVICE void run(const Params& p)
{
    using G = Geometry<Rows>;
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
    const uint32_t a_row = (uint32_t)wave * 16u + lane / 4u;
    const uint32_t a_index = a_row < (uint32_t)p.m ? a_row : 0xffffffffu;
    const uint32_t chunk = (lane % 4u) ^ ((a_row >> 2) & 3u);
    const uint32_t a_offset = a_row * (pitch - stride) + chunk * 16u;
    const uint64_t a_pair = (uint64_t)a_index | ((uint64_t)a_offset << 32);
    const uint32_t b_row = (uint32_t)threadIdx.x >> 4;
    const uint32_t b_octet = (lane & 15u) ^ ((b_row & 1u) * 4u);
    const uint32_t b_offset =
        (b_row * (uint32_t)p.ldb + (uint32_t)p.n0 + b_octet * 8u) * 2u;
    F32x4 c[G::MFragments][G::NFragments];
    static_for<0, G::MFragments * G::NFragments, 1>{}([&](auto ci) {
        constexpr int i = decltype(ci)::value;
        c[i / G::NFragments][i % G::NFragments] = F32x4{0, 0, 0, 0};
    });

    // Publish/read operands zero and one before either plane is reused.
    prefetch_a<Rows, 0>(a_srd, a_pair, wave, 0);
    prefetch_b<Rows, 0>(b_srd, b_offset, wave, p.ldb, 0);
    prefetch_a<Rows, 1>(a_srd, a_pair, wave, 32);
    prefetch_b<Rows, 1>(b_srd, b_offset, wave, p.ldb, 32);
    prefetch_b<Rows, 2>(b_srd, b_offset, wave, p.ldb, 64);
    asm volatile("s_waitcnt vmcnt(2)\n\ts_barrier\n" ::: "memory");
    Operands<Rows> operand[3];
    read<Rows, 0, 0>(operand[0], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    read<Rows, 1, 1>(operand[1], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
    prefetch_a<Rows, 0>(a_srd, a_pair, wave, 64);
    prefetch_b<Rows, 0>(b_srd, b_offset, wave, p.ldb, 96);

    const int stages = p.k / 32;
    int base = 0;
    for(; base + G::Cycle + 4 < stages; base += G::Cycle)
    {
        static_for<0, G::Cycle, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            if constexpr(ComputeFirst) coop::calculate(operand[j % 3], c);
            ready<2>();
            read<Rows, (j + 2) % 2, (j + 2) % 3>(operand[(j + 2) % 3], lane, wave);
            prefetch_a<Rows, (j + 3) % 2>(a_srd, a_pair, wave, (base + j + 3) * 32);
            prefetch_b<Rows, (j + 4) % 3>(b_srd, b_offset, wave, p.ldb,
                                        (base + j + 4) * 32);
            if constexpr(!ComputeFirst) coop::calculate(operand[j % 3], c);
        });
    }
    // At most Cycle+4 real stages remain. Never prefetch beyond the actual K.
    static_for<0, G::Cycle + 5, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        const int t = base + j;
        if(t < stages)
        {
            if constexpr(ComputeFirst) coop::calculate(operand[j % 3], c);
            if(t + 3 < stages) ready<2>();
            else ready<0>();
            if(t + 2 < stages)
                read<Rows, (j + 2) % 2, (j + 2) % 3>(operand[(j + 2) % 3], lane, wave);
            if(t + 3 < stages)
                prefetch_a<Rows, (j + 3) % 2>(a_srd, a_pair, wave, (t + 3) * 32);
            if(t + 4 < stages)
                prefetch_b<Rows, (j + 4) % 3>(b_srd, b_offset, wave, p.ldb, (t + 4) * 32);
            if constexpr(!ComputeFirst) coop::calculate(operand[j % 3], c);
        }
    });
    // Last real stage executed ready<0> before computing. All global producers
    // and LDS readers are retired and every wave crossed the reuse barrier.
    // C owns the full K dimension; the existing store only rearranges output.
    coop::store<Rows, 32, 4>(p, c, lane, wave);
}
} // namespace compact_detail

namespace wide_detail {
using namespace decode_nn_cooperative_wide;
template<int Rows, int BlockK, int Depth, bool ComputeFirst>
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
    const uint32_t chunk = (lane % Chunks) ^ ((a_row >> 2) & (Chunks - 1u));
    const uint32_t a_offset = a_row * (pitch - stride) + chunk * 16u;
    const uint64_t a_pair = (uint64_t)a_index | ((uint64_t)a_offset << 32);
    const uint32_t b_row = (uint32_t)threadIdx.x >> 4;
    const uint32_t b_octet = (lane & 15u) ^ ((b_row & 1u) * 4u);
    const uint32_t b_offset = (b_row * (uint32_t)p.ldb + (uint32_t)p.n0 + b_octet * 8u) * 2u;
    F32x4 c[G::MFragments][G::NFragments];
    static_for<0, G::MFragments * G::NFragments, 1>{}([&](auto ci) {
        constexpr int i = decltype(ci)::value;
        c[i / G::NFragments][i % G::NFragments] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        prefetch<Rows, BlockK, Depth, j>(a_srd, a_pair, b_srd, b_offset, wave, p.ldb, j * BlockK);
    });
    decode_nn_cooperative_wide::ready<Rows, BlockK, Depth, Depth - 2>(wave);
    asm volatile("s_barrier\n" ::: "memory");
    Operands<Rows, BlockK, Depth> operand[3];
    read<Rows, BlockK, Depth, 0>(operand[0], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    read<Rows, BlockK, Depth, 1>(operand[1], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
    prefetch<Rows, BlockK, Depth, 0>(a_srd, a_pair, b_srd, b_offset, wave, p.ldb, Depth * BlockK);
    constexpr int Cycle = 3 * Depth;
    const int stages = p.k / BlockK;
    int base = 0;
    for(; base + Cycle + Depth < stages; base += Cycle)
    {
        static_for<0, Cycle, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            if constexpr(ComputeFirst) calculate(operand[j % 3], c);
            decode_nn_cooperative_wide::ready<Rows, BlockK, Depth, Depth - 2>(wave);
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
            read<Rows, BlockK, Depth, (j + 2) % Depth>(operand[(j + 2) % 3], lane, wave);
            prefetch<Rows, BlockK, Depth, (j + 1) % Depth>(a_srd, a_pair, b_srd,
                b_offset, wave, p.ldb, (base + j + Depth + 1) * BlockK);
            if constexpr(!ComputeFirst) calculate(operand[j % 3], c);
        });
    }
    // Tail is bounded by Cycle+Depth. Publish only real stages, never an OOB
    // speculative panel, and drain every reader before LDS becomes scratch.
    static_for<0, Cycle + Depth + 1, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        const int t = base + j;
        if(t < stages)
        {
            if constexpr(ComputeFirst) calculate(operand[j % 3], c);
            if(t + 2 < stages)
            {
                const int remaining = stages - t - 3;
                const int pending = remaining < Depth - 2 ? remaining : Depth - 2;
                tail_ready<Rows, BlockK, Depth, Depth - 2>(wave, pending);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
                read<Rows, BlockK, Depth, (j + 2) % Depth>(operand[(j + 2) % 3], lane, wave);
            }
            else
            {
                decode_nn_cooperative_wide::ready<Rows, BlockK, Depth, 0>(wave);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
            }
            if(t + Depth + 1 < stages)
                prefetch<Rows, BlockK, Depth, (j + 1) % Depth>(a_srd, a_pair, b_srd,
                    b_offset, wave, p.ldb, (t + Depth + 1) * BlockK);
            if constexpr(!ComputeFirst) calculate(operand[j % 3], c);
        }
    });
    store<Rows, BlockK, Depth>(p, c, lane, wave);
}
} // namespace wide_detail

template<int Rows>
CK_TILE_DEVICE void run_compact(const decode_nn_mainline::Params& p)
{
    compact_detail::run<Rows, true>(p);
}
template<int Rows>
CK_TILE_DEVICE void run_wide(const decode_nn_mainline::Params& p)
{
    wide_detail::run<Rows, 32, 4, true>(p);
}
} // namespace decode_nn_resource_phase
} // namespace ck_tile
#pragma clang diagnostic pop
