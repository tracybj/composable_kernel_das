// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_n256.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
namespace ck_tile { namespace decode_nn_cooperative_n128_noslc {
using namespace decode_nn_mainline;
CK_TILE_DEVICE void load_a_lds(Srd srd, uint64_t pair, uint32_t offset, uint32_t base)
{
    asm volatile("s_mov_b32 m0, %3\n\t"
                 "buffer_load_dwordx4 %0, %1, %2 idxen offen lds\n"
                 :: "v"(pair), "s"(srd), "s"(offset), "s"(base) : "memory");
}

using decode_nn_shared::read_a;
template<int Rows,int BlockK,int Depth,bool BReuse=false,int NTile=256,bool UniformWaves=false> struct Geometry {
 static_assert(Rows==1||Rows==2||Rows==4);
 static_assert(BlockK==32&&(Depth==2||Depth==3));
 static_assert(NTile==256 || (NTile==128 && Depth==2));
 static_assert(!UniformWaves || NTile==128);
 static constexpr int TileN=NTile,Waves=NTile==128?((UniformWaves||Rows==1)?4:8):8;
 static constexpr int Halves=2,ALoadRows=16,APanels=Rows;
 static constexpr int APanelsPerWave=(APanels+Waves-1)/Waves;
 static constexpr int ABytes=APanels*1024;
 static constexpr int StageBytes=ABytes+TileN*64,LdsBytes=Depth*StageBytes;
 // The original partition stays the default for every existing instance.
 static constexpr int MGroups=NTile==128?((UniformWaves||Rows==1)?1:2):((BReuse && Rows>1)?Rows/2:Rows),NGroups=Waves/MGroups;
 static constexpr int BLoadsPerWave=TileN*64/(Waves*1024),BRowsPerLoad=Waves*1024/(TileN*2);
 static constexpr int MFragments=Rows/MGroups,WaveM=MFragments*16;
 static constexpr int WaveN=TileN/NGroups,NFragments=WaveN/16,BReads=NFragments/2;
 static_assert(LdsBytes<=65536 && NFragments>=2 && NFragments%2==0);
};
template<int Rows,int BlockK,int Depth,bool BReuse=false,int NTile=256,bool UniformWaves=false> struct Operands {
 using G=Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
 Bf16x4 a[G::Halves][G::MFragments];
 Bf16x8 b[G::Halves][G::BReads];
};
template<int Rows,int BlockK,int Depth,int Slot,bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void prefetch(Srd as,const uint64_t (&pairs)[Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::APanelsPerWave],Srd bs,uint32_t bo,int wave,int ldb,int k) {
 using G=Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
 static_for<0,G::APanelsPerWave,1>{}([&](auto ai) {
  constexpr int i=decltype(ai)::value;
  if(wave+i*G::Waves<G::APanels)
   load_a_lds(as,pairs[i],(uint32_t)k*2u,Slot*G::StageBytes+(uint32_t)(wave+i*G::Waves)*1024u);
 });
 static_for<0,G::BLoadsPerWave,1>{}([&](auto hi) {
  constexpr int h=decltype(hi)::value;
  load_b(bs,bo,(uint32_t)(k+h*G::BRowsPerLoad)*(uint32_t)ldb*2u,Slot*G::StageBytes+G::ABytes+h*G::Waves*1024u+(uint32_t)wave*1024u);
 });
}
template<int Rows,int BlockK,int Depth,int Pending,bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void ready(int wave) {
 static_assert(Pending>=0 && Pending<=Depth-2);
 if constexpr(Pending==0)
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
 else if(wave<Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::APanels)
  asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"((Geometry<Rows,BlockK,Depth,BReuse,NTile,UniformWaves>::BLoadsPerWave+1)*Pending) : "memory");
 else
  asm volatile("s_waitcnt vmcnt(%0)\n" :: "n"(Geometry<Rows,BlockK,Depth,BReuse,NTile,UniformWaves>::BLoadsPerWave*Pending) : "memory");
}
template<int Rows, int BlockK, int Depth, bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void store(const Params& p,
    F32x4 (&c)[Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::MFragments]
               [Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::NFragments], uint32_t lane, int wave)
{
    using G = Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
    const uint32_t wave_m = (uint32_t)wave / G::NGroups;
    const uint32_t wave_n = (uint32_t)wave % G::NGroups;
    const uint32_t base = (uint32_t)wave * 1024u;
    const uint32_t row_lane = lane & 15u;
    const uint32_t q = lane >> 4;
    static_for<0, G::MFragments, 1>{}([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<0, G::BReads, 1>{}([&](auto ri) {
            constexpr int r = decltype(ri)::value;
            static_for<0, 8, 1>{}([&](auto ei) {
                constexpr int e = decltype(ei)::value;
                const uint32_t byte = (q + (uint32_t)((e / 4) * 16 + (e % 4) * 4)) * 2u;
                const uint32_t address = base + row_lane * 64u +
                    (((byte >> 4) ^ (row_lane & 3u)) << 4) + (byte & 15u);
                const uint32_t value = to_bf16(c[m][2 * r + e / 4][e % 4]);
                asm volatile("ds_write_b16 %0, %1\n" :: "v"(address), "v"(value) : "memory");
            });
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            const uint32_t row = lane >> 2;
            const uint32_t unit = lane & 3u;
            const uint32_t address = base + row * 64u + ((unit ^ (row & 3u)) << 4);
            U32x4 packed;
            asm volatile("ds_read_b128 %0, %1\n" : "=v"(packed) : "v"(address) : "memory");
            asm volatile("s_waitcnt lgkmcnt(0)\n" : "+v"(packed) :: "memory");
            const uint32_t output_m = wave_m * G::WaveM + m * 16u + row;
            if(output_m < (uint32_t)p.m)
                *reinterpret_cast<U32x4*>(p.c + (int64_t)output_m * p.ldc + p.n0 +
                    wave_n * G::WaveN + r * 32u + unit * 8u) = packed;
        });
    });
}

template<int Rows,int BlockK,int Depth,int Pending,bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void tail_ready(int wave,int pending) {
 if constexpr(Pending==0) ready<Rows,BlockK,Depth,0,BReuse,NTile,UniformWaves>(wave);
 else if(pending==0) ready<Rows,BlockK,Depth,0,BReuse,NTile,UniformWaves>(wave);
 else ready<Rows,BlockK,Depth,Pending,BReuse,NTile,UniformWaves>(wave);
}
template<int Rows, int BlockK, int Depth, int Slot, bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void read(Operands<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>& operand,
                         uint32_t lane, int wave)
{
    using G = Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
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
            operand.b[h][r] = read_b(Slot * G::StageBytes + G::ABytes + h * G::TileN * 32u +
                row * G::TileN * 2u + ((octet ^ ((row & 1u) * 4u)) * 16u));
        });
    });
}

template<int Rows, int BlockK, int Depth, bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void calculate(const Operands<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>& operand,
    F32x4 (&c)[Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::MFragments]
               [Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>::NFragments])
{
    using G = Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
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

template<int Rows, int BlockK, int Depth, bool ComputeFirst, bool BReuse=false,int NTile=256,bool UniformWaves=false>
CK_TILE_DEVICE void run_impl(const Params& p)
{
    using G = Geometry<Rows, BlockK, Depth, BReuse,NTile,UniformWaves>;
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
        const uint32_t row=((uint32_t)wave+i*G::Waves)*16u+lane/4u;
        const uint32_t index=row<(uint32_t)p.m?row:0xffffffffu;
        const uint32_t chunk=(lane%4u)^((row>>2)&3u);
        const uint32_t offset=row*(pitch-stride)+chunk*16u;
        a_pair[i]=(uint64_t)index|((uint64_t)offset<<32);
    });
    const uint32_t b_row = (uint32_t)threadIdx.x / (G::TileN/8u);
    const uint32_t b_octet = (lane & (G::TileN/8u-1u)) ^ ((b_row & 1u) * 4u);
    const uint32_t b_offset = (b_row * (uint32_t)p.ldb + (uint32_t)p.n0 + b_octet * 8u) * 2u;
    F32x4 c[G::MFragments][G::NFragments];
    static_for<0, G::MFragments * G::NFragments, 1>{}([&](auto ci) {
        constexpr int i = decltype(ci)::value;
        c[i / G::NFragments][i % G::NFragments] = F32x4{0, 0, 0, 0};
    });
    static_for<0, Depth, 1>{}([&](auto si) {
        constexpr int j = decltype(si)::value;
        prefetch<Rows, BlockK, Depth, j,BReuse,NTile,UniformWaves>(a_srd, a_pair, b_srd, b_offset, wave, p.ldb, j * BlockK);
    });
    ready<Rows, BlockK, Depth, Depth - 2,BReuse,NTile,UniformWaves>(wave);
    asm volatile("s_barrier\n" ::: "memory");
    Operands<Rows, BlockK, Depth, BReuse,NTile,UniformWaves> operand[3];
    read<Rows, BlockK, Depth, 0,BReuse,NTile,UniformWaves>(operand[0], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    read<Rows, BlockK, Depth, 1,BReuse,NTile,UniformWaves>(operand[1], lane, wave);
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
    prefetch<Rows, BlockK, Depth, 0,BReuse,NTile,UniformWaves>(a_srd, a_pair, b_srd, b_offset, wave, p.ldb, Depth * BlockK);
    constexpr int Cycle = 3 * Depth;
    const int stages = p.k / BlockK;
    int base = 0;
    for(; base + Cycle + Depth < stages; base += Cycle)
    {
        static_for<0, Cycle, 1>{}([&](auto si) {
            constexpr int j = decltype(si)::value;
            if constexpr(ComputeFirst) calculate(operand[j % 3], c);
            ready<Rows, BlockK, Depth, Depth - 2,BReuse,NTile,UniformWaves>(wave);
            asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
            read<Rows, BlockK, Depth, (j + 2) % Depth,BReuse,NTile,UniformWaves>(operand[(j + 2) % 3], lane, wave);
            prefetch<Rows, BlockK, Depth, (j + 1) % Depth,BReuse,NTile,UniformWaves>(a_srd, a_pair, b_srd,
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
                tail_ready<Rows, BlockK, Depth, Depth - 2,BReuse,NTile,UniformWaves>(wave, pending);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
                read<Rows, BlockK, Depth, (j + 2) % Depth,BReuse,NTile,UniformWaves>(operand[(j + 2) % 3], lane, wave);
            }
            else
            {
                ready<Rows, BlockK, Depth, 0,BReuse,NTile,UniformWaves>(wave);
                asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
            }
            if(t + Depth + 1 < stages)
                prefetch<Rows, BlockK, Depth, (j + 1) % Depth,BReuse,NTile,UniformWaves>(a_srd, a_pair, b_srd,
                    b_offset, wave, p.ldb, (t + Depth + 1) * BlockK);
            if constexpr(!ComputeFirst) calculate(operand[j % 3], c);
        }
    });
    store<Rows,BlockK,Depth,BReuse,NTile,UniformWaves>(p,c,lane,wave);
}

template<int Rows,int Depth=2,bool BReuse=false,int NTile=256,bool UniformWaves=false> CK_TILE_DEVICE void run(const Params& p) {run_impl<Rows,32,Depth,true,BReuse,NTile,UniformWaves>(p);}
} }
#pragma clang diagnostic pop
