// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_wide.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile { namespace decode_nn_cooperative_k64_wide {
using namespace decode_nn_mainline;
using decode_nn_shared::load_a_lds;
using decode_nn_shared::read_a;
template<int Rows,int BlockK,int Depth> struct Geometry {
 static_assert(Rows==2||Rows==4);
 static_assert(BlockK==64&&Depth==2);
 static constexpr int Halves=4,ALoadRows=8,APanels=Rows*2;
 static constexpr int APanelsPerWave=(APanels+7)/8;
 static constexpr int ABytes=APanels*1024;
 static constexpr int StageBytes=ABytes+16384,LdsBytes=2*StageBytes;
 static constexpr int MGroups=Rows,NGroups=8/MGroups;
 static constexpr int MFragments=Rows/MGroups,WaveM=MFragments*16;
 static constexpr int WaveN=128/NGroups,NFragments=WaveN/16,BReads=NFragments/2;
 static_assert(LdsBytes<=65536);
};
template<int Rows,int BlockK,int Depth> struct Operands {
 using G=Geometry<Rows,BlockK,Depth>;
 Bf16x4 a[G::Halves][G::MFragments];
 Bf16x8 b[G::Halves][G::BReads];
};
template<int Rows,int BlockK,int Depth,int Slot>
CK_TILE_DEVICE void prefetch(Srd as,const uint64_t (&pairs)[Geometry<Rows,BlockK,Depth>::APanelsPerWave],Srd bs,uint32_t bo,int wave,int ldb,int k) {
 using G=Geometry<Rows,BlockK,Depth>;
 static_for<0,G::APanelsPerWave,1>{}([&](auto ai) {
  constexpr int i=decltype(ai)::value;
  if(wave+i*8<G::APanels)
   load_a_lds(as,pairs[i],(uint32_t)k*2u,Slot*G::StageBytes+(uint32_t)(wave+i*8)*1024u);
 });
 static_for<0,2,1>{}([&](auto hi) {
  constexpr int h=decltype(hi)::value;
  load_b(bs,bo,(uint32_t)(k+h*32)*(uint32_t)ldb*2u,Slot*G::StageBytes+G::ABytes+h*8192u+(uint32_t)wave*1024u);
 });
}
template<int Rows,int BlockK,int Depth,int Pending>
CK_TILE_DEVICE void ready(int wave) {
 static_assert(Pending==0);(void)wave;
 asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}
template<int Rows,int BlockK,int Depth,int Pending>
CK_TILE_DEVICE void tail_ready(int wave,int pending) {
 (void)pending;ready<Rows,BlockK,Depth,Pending>(wave);
}
template<int Rows, int BlockK, int Depth, int Slot>
CK_TILE_DEVICE void read(Operands<Rows, BlockK, Depth>& operand,
                         uint32_t lane, int wave)
{
    using G = Geometry<Rows, BlockK, Depth>;
    const uint32_t wave_m = (uint32_t)wave / G::NGroups;
    const uint32_t wave_n = (uint32_t)wave % G::NGroups;
    constexpr int Chunks = BlockK / 8;
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        static_for<0, G::MFragments, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const uint32_t row = wave_m * G::WaveM + m * 16u + (lane & 15u);
            const uint32_t byte = (h * 32u + (lane >> 4) * 8u) ^
                                  (((row >> 2) & (Chunks - 1u)) * 16u);
            operand.a[h][m] = decode_nn_shared::read_a(
                Slot * G::StageBytes + row * BlockK * 2u + byte);
        });
        static_for<0, G::BReads, 1>{}([&](auto ri) {
            constexpr int r = decltype(ri)::value;
            const uint32_t row = lane >> 2;
            const uint32_t octet = wave_n * (G::WaveN / 8u) + r * 4u + (lane & 3u);
            operand.b[h][r] = read_b(Slot * G::StageBytes + G::ABytes + h * 4096u +
                row * 256u + ((octet ^ ((row & 1u) * 4u)) * 16u));
        });
    });
}

template<int Rows, int BlockK, int Depth>
CK_TILE_DEVICE void calculate(const Operands<Rows, BlockK, Depth>& operand,
    F32x4 (&c)[Geometry<Rows, BlockK, Depth>::MFragments]
               [Geometry<Rows, BlockK, Depth>::NFragments])
{
    using G = Geometry<Rows, BlockK, Depth>;
    static_for<0, G::Halves, 1>{}([&](auto hi) {
        constexpr int h = decltype(hi)::value;
        static_for<0, G::MFragments, 1>{}([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<0, G::BReads, 1>{}([&](auto ri) {
                constexpr int r = decltype(ri)::value;
                const auto b = operand.b[h][r];
                const Bf16x4 b0{b[0], b[1], b[2], b[3]};
                const Bf16x4 b1{b[4], b[5], b[6], b[7]};
                mmac(c[m][2 * r], operand.a[h][m], b0);
                mmac(c[m][2 * r + 1], operand.a[h][m], b1);
            });
        });
    });
}

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
    uint64_t a_pair[G::APanelsPerWave];
    static_for<0,G::APanelsPerWave,1>{}([&](auto ai) {
        constexpr int i=decltype(ai)::value;
        const uint32_t row=((uint32_t)wave+i*8u)*8u+lane/8u;
        const uint32_t index=row<(uint32_t)p.m?row:0xffffffffu;
        const uint32_t chunk=(lane%8u)^((row>>2)&7u);
        const uint32_t offset=row*(pitch-stride)+chunk*16u;
        a_pair[i]=(uint64_t)index|((uint64_t)offset<<32);
    });
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
    ready<Rows, BlockK, Depth, Depth - 2>(wave);
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
            ready<Rows, BlockK, Depth, Depth - 2>(wave);
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
                ready<Rows, BlockK, Depth, 0>(wave);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
            }
            if(t + Depth + 1 < stages)
                prefetch<Rows, BlockK, Depth, (j + 1) % Depth>(a_srd, a_pair, b_srd,
                    b_offset, wave, p.ldb, (t + Depth + 1) * BlockK);
            if constexpr(!ComputeFirst) calculate(operand[j % 3], c);
        }
    });
    decode_nn_cooperative_wide::store<Rows,32,4>(p,c,lane,wave);
}

template<int Rows> CK_TILE_DEVICE void run(const Params& p) {run<Rows,64,2,true>(p);}
} }
#pragma clang diagnostic pop
