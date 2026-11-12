// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "gemm_pipeline_decode_nn_cooperative.hpp"

// Experimental resource factor; selected only through explicit decode policy.
// Preserve cooperative complete-M ownership, K32/N128, four waves, and three
// operand banks. Only the global LDS ring changes: two A panels, three B panels.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile {
namespace decode_nn_cooperative_compact {
using namespace decode_nn_mainline;
namespace coop = decode_nn_cooperative;

template<int Rows>
struct Geometry : coop::Geometry<Rows, 32, 4>
{
    using Base = coop::Geometry<Rows, 32, 4>;
    static constexpr int ADepth = 2;
    static constexpr int BDepth = 3;
    static constexpr int BBytes = 8192;
    static constexpr int BBase = ADepth * Base::ABytes;
    static constexpr int LdsBytes = BBase + BDepth * BBytes;
    static constexpr int Cycle = 6;
    static_assert(LdsBytes <= 32768);
};

template<int Rows>
using Operands = coop::Operands<Rows, 32, 4>;

template<int Rows, int Slot>
CK_TILE_DEVICE void prefetch_a(Srd srd, uint64_t pair, int wave, int k)
{
    using G = Geometry<Rows>;
    static_assert(Slot >= 0 && Slot < G::ADepth);
    if(wave < G::ALoadWaves)
        decode_nn_shared::load_a_lds(srd, pair, (uint32_t)k * 2u,
            Slot * G::ABytes + (uint32_t)wave * 1024u);
}

template<int Rows, int Slot>
CK_TILE_DEVICE void prefetch_b(Srd srd, uint32_t offset, int wave, int ldb, int k)
{
    using G = Geometry<Rows>;
    static_assert(Slot >= 0 && Slot < G::BDepth);
    static_for<0, 2, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        load_b(srd, offset, (uint32_t)(k + h * 16) * (uint32_t)ldb * 2u,
            G::BBase + Slot * G::BBytes + h * 4096u + (uint32_t)wave * 1024u);
    });
}

template<int Rows, int ASlot, int BSlot>
CK_TILE_DEVICE void read(Operands<Rows>& operand, uint32_t lane, int wave)
{
    using G = Geometry<Rows>;
    static_assert(ASlot >= 0 && ASlot < G::ADepth);
    static_assert(BSlot >= 0 && BSlot < G::BDepth);
    const uint32_t wave_m = (uint32_t)wave / G::NGroups;
    const uint32_t wave_n = (uint32_t)wave % G::NGroups;
    static_for<0, 2, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        static_for<0, G::MFragments, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const uint32_t row = wave_m * G::WaveM + m * 16u + (lane & 15u);
            const uint32_t byte = (h * 32u + (lane >> 4) * 8u) ^
                                  (((row >> 2) & 3u) * 16u);
            operand.a[h][m] = decode_nn_shared::read_a(
                ASlot * G::ABytes + row * 64u + byte);
        });
        static_for<0, G::BReads, 1>{}([&](auto ri) {
            constexpr int r = decltype(ri)::value;
            const uint32_t row = lane >> 2;
            const uint32_t octet = wave_n * (G::WaveN / 8u) + r * 4u + (lane & 3u);
            operand.b[h][r] = read_b(G::BBase + BSlot * G::BBytes + h * 4096u +
                row * 256u + ((octet ^ ((row & 1u) * 4u)) * 16u));
        });
    });
}

template<int Pending>
CK_TILE_DEVICE void ready()
{
    static_assert(Pending == 0 || Pending == 2);
    // The youngest future B panel is exactly two VMEM instructions per wave.
    // A is issued before that B panel. Retaining only those two instructions
    // retires all required older A/B producers, including waves not loading A.
    // Real ISA order and GPU correctness still require validation.
    asm volatile("s_waitcnt vmcnt(%0)\n\ts_waitcnt lgkmcnt(0)\n\ts_barrier\n"
                 :: "n"(Pending) : "memory");
}

template<int Rows>
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
            ready<2>();
            read<Rows, (j + 2) % 2, (j + 2) % 3>(operand[(j + 2) % 3], lane, wave);
            prefetch_a<Rows, (j + 3) % 2>(a_srd, a_pair, wave, (base + j + 3) * 32);
            prefetch_b<Rows, (j + 4) % 3>(b_srd, b_offset, wave, p.ldb,
                                        (base + j + 4) * 32);
            coop::calculate(operand[j % 3], c);
        });
    }
    // At most Cycle+4 real stages remain. Never prefetch beyond the actual K.
    static_for<0, G::Cycle + 5, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        const int t = base + j;
        if(t < stages)
        {
            if(t + 3 < stages) ready<2>();
            else ready<0>();
            if(t + 2 < stages)
                read<Rows, (j + 2) % 2, (j + 2) % 3>(operand[(j + 2) % 3], lane, wave);
            if(t + 3 < stages)
                prefetch_a<Rows, (j + 3) % 2>(a_srd, a_pair, wave, (t + 3) * 32);
            if(t + 4 < stages)
                prefetch_b<Rows, (j + 4) % 3>(b_srd, b_offset, wave, p.ldb, (t + 4) * 32);
            coop::calculate(operand[j % 3], c);
        }
    });
    // Last real stage executed ready<0> before computing. All global producers
    // and LDS readers are retired and every wave crossed the reuse barrier.
    // C owns the full K dimension; the existing store only rearranges output.
    coop::store<Rows, 32, 4>(p, c, lane, wave);
}
} // namespace decode_nn_cooperative_compact
} // namespace ck_tile
#pragma clang diagnostic pop
