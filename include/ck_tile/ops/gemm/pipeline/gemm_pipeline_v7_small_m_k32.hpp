// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Experimental 64x{128,64}x32: own K permutation, four-stage LDS ring.
// N128:48KB, three D2L/six DS per stage; N64:32KB, two D2L/four DS.
// Each K32 stage performs two K16 MMAC substeps.
// First version intentionally retains the existing redundant-load K tail,
// so geometry and no-load drain can be evaluated independently.
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
namespace ck_tile {
template <index_t TileN>
struct V7SmallMK32Geometry
{
    static constexpr index_t kBlockM=64, kBlockN=TileN, kStageK=32;
    static constexpr index_t kWaveSize=64, kWaves=4, kThreads=256;
    static constexpr index_t kWaveTileM=32, kWaveTileN=TileN/2;
    static constexpr index_t kMTilesPerWave=2, kNTilesPerWave=TileN/32;
    static constexpr index_t kMmacPerStage=2*kNTilesPerWave; // accumulator fragments; two MMACs per fragment/stage
    static constexpr index_t kReadsPerStage=2+2*(TileN/64);
    static constexpr index_t kLoadsPerStage=1+TileN/64;
    static constexpr index_t kNumStages=4, kOpRing=3, kPhaseCycle=12, kWaveGroup=2, kPrefetchDist=5;
    static constexpr index_t kLdsBytesA=4096, kLdsBytesB=32*TileN*2, kLdsBytesPerStage=kLdsBytesA+kLdsBytesB;
    static constexpr index_t kLdsBytes=kNumStages*kLdsBytesPerStage;
};
namespace v7k32 {
template <typename G, uint32_t Offset>
CK_TILE_DEVICE void read_a(const uint32_t (&base)[2],
                            v7u::vec4_bf16 (&out)[2 * G::kMTilesPerWave])
{
    v7u::vec8_bf16 d0,d1;
    asm volatile("ds_read_b128 %0, %2 offset:%4\n\t"
                 "ds_read_b128 %1, %3 offset:%4\n"
        : "=&v"(d0), "=&v"(d1) : "v"(base[0]), "v"(base[1]), "n"(Offset) : "memory");
    out[0]=__builtin_shufflevector(d0,d0,0,1,2,3);
    out[1]=__builtin_shufflevector(d1,d1,0,1,2,3);
    out[2]=__builtin_shufflevector(d0,d0,4,5,6,7);
    out[3]=__builtin_shufflevector(d1,d1,4,5,6,7);
}
template <typename G, uint32_t Offset>
CK_TILE_DEVICE void read_b(const uint32_t (&base)[G::kNTilesPerWave/2],
                            v7u::vec4_bf16 (&out)[2 * G::kNTilesPerWave])
{
    constexpr uint32_t half_stride=16u * G::kBlockN * 2u;
    if constexpr(G::kNTilesPerWave == 4)
    {
        v7u::vec4_bf16 half0[4],half1[4];
        v7u::ds_read_m32x16_x4<Offset>(base,half0);
        v7u::ds_read_m32x16_x4<Offset+half_stride>(base,half1);
#pragma unroll
        for(int i=0;i<4;++i) { out[i]=half0[i]; out[i+4]=half1[i]; }
    }
    else
    {
        v7u::vec8_bf16 h0,h1;
        asm volatile("ds_read_m32x16_b16 %0, %2 offset:%3\n\t"
                     "ds_read_m32x16_b16 %1, %2 offset:%4\n"
            : "=&v"(h0), "=&v"(h1)
            : "v"(base[0]), "n"(Offset), "n"(Offset+half_stride) : "memory");
        out[0]=__builtin_shufflevector(h0,h0,0,1,2,3);
        out[1]=__builtin_shufflevector(h0,h0,4,5,6,7);
        out[2]=__builtin_shufflevector(h1,h1,0,1,2,3);
        out[3]=__builtin_shufflevector(h1,h1,4,5,6,7);
    }
}
} // namespace v7k32
template <typename Problem, index_t TileN>
struct GemmPipelineV7SmallMK32
{
    using G       = V7SmallMK32Geometry<TileN>;
    static_assert(TileN == 128 || TileN == 64);
    using AccTile = V7SmallMAccTile<G>;
    static constexpr V7UniformKind kKind = V7UniformKind::NN;

    // The framework's trait for the raw (no tensor view, no tile window) kernel
    // path: see GroupedGemmKernel's is_grouped_gemm_v7_uniform_pipeline.
    static constexpr bool IsV7UniformPipeline = true;

    static constexpr V7UniformKind kind = kKind;

    // ---- framework-facing surface (what GroupedGemmKernel reads) --------
    // The kernel takes BlockSize as a VALUE, not a call: keep the member form.
    static constexpr index_t BlockSize = G::kThreads;

    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;
    using ALayout   = remove_cvref_t<typename Problem::ALayout>;
    using BLayout   = remove_cvref_t<typename Problem::BLayout>;
    using CLayout   = remove_cvref_t<typename Problem::CLayout>;

    static constexpr index_t KPerBlock = remove_cvref_t<typename Problem::BlockGemmShape>::kK;

    // The one-CTA-per-tile structure the register budget depends on: the config
    // must be non-persistent, and nothing double-buffers.
    static constexpr bool UsePersistentKernel = false;
    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr bool Preshuffle          = false;

    // The 32 KB ring permits two resident CTAs when registers allow.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return G::kLdsBytes; }

    using AOpBank = v7u::vec4_bf16[G::kOpRing][2 * G::kMTilesPerWave];
    using BOpBank = v7u::vec4_bf16[G::kOpRing][2 * G::kNTilesPerWave];

    // Per-thread scalar state: the descriptors / per-lane bases the k loop needs.
    struct Addr
    {
        v7u::buffer_rsrc_t a_rsrc, b_rsrc;
        uint32_t a_vaddr, b_vaddr;      // per-lane global byte offsets (TN, NN-B)
        uint64_t b_pair;
        uint64_t a_pair;                // NN A: {index, offset} idxen pair
        uint32_t a_rd[2];                  // NN A b64 read base
        uint32_t a_rb[2];               // TN A m32x16 read bases
        uint32_t b_rb[G::kNTilesPerWave / 2];               // B m32x16 read bases
        uint32_t m0_a_base, m0_b_base;
        int b_stride_bytes;
        int a_stride_bytes;   // TN: the A panel's row pitch (the k advance);
                              // NN keeps the K-contiguous 2-byte advance and
                              // never reads this.
        int n_stages;
        int wid;
    };

    CK_TILE_DEVICE static void mmac_tile(AccTile& acc,
                                         const AOpBank& a_bank,
                                         const BOpBank& b_bank,
                                         int bank)
    {
        // Two complementary 16-term subsets of each 32-wide K panel.
        // Each lane's A b128 splits into the two matching four-BF16 operands.
#pragma unroll
        for(index_t i = 0; i < 2 * G::kMmacPerStage; ++i)
        {
            const index_t half = i / G::kMmacPerStage;
            const index_t tile = i % G::kMmacPerStage;
            const index_t m = tile / G::kNTilesPerWave;
            const index_t n = tile % G::kNTilesPerWave;
            acc.v[tile] = v7u::mmac_bf16(a_bank[bank][half * G::kMTilesPerWave + m],
                                       b_bank[bank][half * G::kNTilesPerWave + n],
                                       acc.v[tile]);
            if(i == 0) v7u::set_prio<3>();
        }
        v7u::set_prio<1>();
    }

    template <int kBank, uint32_t kReadSlot, uint32_t kLoadSlot>
    CK_TILE_DEVICE static void issue_load_and_read(Addr& s,
                                                   AOpBank& a_bank,
                                                   BOpBank& b_bank,
                                                   int kt)
    {
        if((kt + 2) < s.n_stages)
        {
            if constexpr(kKind == V7UniformKind::NN)
            {
                v7k32::read_b<G, kReadSlot>(s.b_rb, b_bank[kBank]);
                v7k32::read_a<G, kReadSlot>(s.a_rd, a_bank[kBank]);
            }
            else
            {
                v7k32::read_b<G, kReadSlot>(s.b_rb, b_bank[kBank]);
                v7u::ds_read_m32x16_x4<kReadSlot>(s.a_rb, a_bank[kBank]);
            }
        }
        // The k advance rides the soffsets in every form: the descriptor base is
        // never rewritten inside the loop.
        const int kStage   = (kt + G::kPrefetchDist < s.n_stages) ? kt + G::kPrefetchDist : 0;
        // NN advances one32-wide K stage (64 bytes) in each row of A.
        // B advances32 physical rows; its two panels use the matched K permutation.
        const uint32_t so_a =
            (uint32_t)(kStage * G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)s.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(kStage * G::kStageK * s.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            v7u::buffer_load_dwordx4_idxen_lds_t(
                s.a_rsrc, s.a_pair, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_idxen_lds(
                s.b_rsrc, s.b_pair, so_b, s.m0_b_base + kLoadSlot);
            if constexpr(G::kBlockN == 128)
            {
            v7u::buffer_load_dwordx4_idxen_lds(
                s.b_rsrc, s.b_pair, so_b + 4u * (uint32_t)s.b_stride_bytes,
                s.m0_b_base + kLoadSlot + 4096u);
            }
        }
        else
        {
            v7u::buffer_load_dwordx4_offen_lds(
                s.a_rsrc, s.a_vaddr, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_idxen_lds(
                s.b_rsrc, s.b_pair, so_b, s.m0_b_base + kLoadSlot);
            if constexpr(G::kBlockN == 128)
            {
            v7u::buffer_load_dwordx4_idxen_lds(
                s.b_rsrc, s.b_pair, so_b + 4u * (uint32_t)s.b_stride_bytes,
                s.m0_b_base + kLoadSlot + 4096u);
            }
        }
    }

    // One k-stage. kGroupA selects the phase order (compile-time: the divergent
    // runtime branch per step compiled to a full exec-mask prologue every step);
    // kBarrier=false on the last step of an unrolled cycle, where the caller
    // emits the barrier after its loop control so the trip test overlaps the
    // drain instead of being exposed behind it.
    template <int kPos,
              bool kGroupA  = false,
              bool kBarrier = true,
              int kBankCur  = kPos % G::kOpRing,
              uint32_t kBankNext = (kPos + 2) % G::kOpRing,
              uint32_t kReadSlot = (uint32_t)(((kPos + 2) % G::kNumStages) * G::kLdsBytesPerStage),
              uint32_t kLoadSlot =
                  (uint32_t)(((kPos + G::kPrefetchDist) % G::kNumStages) * G::kLdsBytesPerStage)>
    CK_TILE_DEVICE static void stage_step(Addr& s, AccTile& acc, AOpBank& a_bank, BOpBank& b_bank, int kt)
    {
        if constexpr(kGroupA)
        {
            mmac_tile(acc, a_bank, b_bank, kBankCur);
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(G::kReadsPerStage);
        }
        else
        {
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(G::kReadsPerStage);
            mmac_tile(acc, a_bank, b_bank, kBankCur);
        }
        v7u::wait_all_buffer_data_arrived<false>(2 * G::kLoadsPerStage);
        // Retire the look-ahead reads before the block barrier: on the next
        // step a different wave can overwrite that shared ring slot.
        v7u::wait_lds_data_arrived<false>(0);
        if constexpr(kBarrier)
        {
            v7u::wait_all_warp_arrived();
        }
    }

    CK_TILE_DEVICE static void run_tile(const V7UniformTileParams& p, AccTile& acc, void* smem);
};

// ---------------------------------------------------------------------------
// The tile body. Written once, with `if constexpr` on the operand layout; the
// ring arithmetic follows the original V7 kernel. The smaller MMAC bursts
// require the additional LDS completion wait before each block barrier.
// ---------------------------------------------------------------------------
template <typename Problem, index_t TileN>
CK_TILE_DEVICE void GemmPipelineV7SmallMK32<Problem, TileN>::run_tile(
    const V7UniformTileParams& p, AccTile& acc, void* smem)
{
    using namespace v7u_detail;
    (void)smem;   // the ring is the launcher's dynamic LDS allocation

    const index_t tid  = threadIdx.x;
    const int wid      = (int)(tid >> 6);
    const int lane     = (int)(tid & (G::kWaveSize - 1));
    const int wave_m   = wid >> 1;
    const int wave_n   = wid & 1;
    const int wave_m_start = wave_m * (int)G::kWaveTileM;

    const int m_block_start = p.m_start;
    const int n_block_start = p.n_start;

#pragma unroll
    for(index_t i = 0; i < G::kMmacPerStage; ++i) acc.v[i] = vec4_fp32{0, 0, 0, 0};

    Addr S;
    S.wid            = wid;
    S.n_stages       = (int)(p.k / G::kStageK);
    S.b_stride_bytes = (int)p.b_stride * 2;

    const uint32_t m0_a_base = __builtin_amdgcn_readfirstlane((uint32_t)(wid * 1024));
    const uint32_t m0_b_base =
        __builtin_amdgcn_readfirstlane((uint32_t)(G::kLdsBytesA + wid * 1024));
    S.m0_a_base = m0_a_base;
    S.m0_b_base = m0_b_base;

    if constexpr(kKind == V7UniformKind::NN)
    {
        // ---- A = [M, K], K-contiguous ------------------------------------
        const int a_stride_bytes = (int)p.a_stride * 2;
        // Lane pairs fetch one row's two 16 B k-halves; rows r = 4..7, 12..15
        // (mod 16) take the halves swapped, which spreads the b64 read phases
        // over all 8 bank groups (a plain row-major 32 B stride reaches 4 and
        // measured 2-way conflicts on every read phase).
        const int a_row_global = m_block_start + wid * 16 + (lane >> 2);
        // Four 16-byte lanes cover one complete 32-wide A row. Rotate
        // quarters by the row group; the LDS read below reverses the rotation.
        const uint32_t a_kh = (uint32_t)(lane & 3) ^ (uint32_t)((lane >> 4) & 3);
        // Descriptor STRIDE = rotation period: the largest power-of-two multiple
        // of 64 that fits the 14-bit field. Non-power-of-two pitches keep the
        // rotation through the idxen split (index*STRIDE + row*rem + col is
        // byte-identical to row*pitch + col).
        S.a_stride_bytes    = a_stride_bytes;   // unused on the NN advance path
        uint32_t a_stride_desc = 64u;
        while(a_stride_desc * 2u <= (uint32_t)a_stride_bytes && a_stride_desc * 2u <= 0x3fffu)
        {
            a_stride_desc *= 2u;
        }
        const uint32_t a_off_rem = (uint32_t)a_stride_bytes - a_stride_desc;
        // Rows past this group's M must not be fetched: the strided idxen
        // descriptor range-checks the INDEX, and its size field (bytes) does
        // not bound the row count, so an M tail would read up to 255 rows past
        // the group. Index 0xffffffff is out of range for any size: the load
        // writes 0 into the row's LDS slot without a memory access (measured).
        const uint32_t a_index =
            a_row_global < p.m && a_row_global < m_block_start + G::kBlockM ? (uint32_t)a_row_global : 0xffffffffu;
        S.a_pair = (uint64_t)a_index |
                   ((uint64_t)((uint32_t)(a_row_global * (int)a_off_rem + a_kh * 16)) << 32);
        const uint32_t a_size_bytes = (uint32_t)((int64_t)p.m * a_stride_bytes);
        S.a_rsrc = make_buffer_rsrc(p.a_ptr, a_stride_desc, a_size_bytes, /*swizzle=*/true);
        // Read base: row (lane&15) at +32*row, 8 B k-unit (lane>>4) at +du, with
        // du compensating the swapped rows' [kh1|kh0] placement.
        const uint32_t a_rho = (uint32_t)(lane & 15);
        const uint32_t a_uu = (uint32_t)(lane >> 4);
        const uint32_t a_rotation = ((a_rho >> 2) & 3u) * 16u;
        S.a_rd[0] = (uint32_t)(wave_m_start * 64) + a_rho * 64u +
                    ((a_uu * 16u) ^ a_rotation);
        S.a_rd[1] = S.a_rd[0] + 1024u; // second 16-row fragment

    }
    else
    {
        // ---- A = [K, M], M-contiguous (the mirrored NN-B case) ------------
        const int a_stride_bytes = (int)p.a_stride * 2;
        const int a_k_in_stage   = wid * 2 + (lane >> 5);
        const int a_m_global =
            m_block_start + 8 * (((lane >> 3) & 3) * 8 + ((lane & 7) ^ (4 * ((lane >> 5) & 1))));
        S.a_vaddr = (uint32_t)(a_k_in_stage * a_stride_bytes + a_m_global * 2);
        const uint32_t a_size_bytes = (uint32_t)((int64_t)p.k * a_stride_bytes);
        S.a_rsrc = make_buffer_rsrc(p.a_ptr, (uint32_t)a_stride_bytes, a_size_bytes, false);
        S.a_stride_bytes = a_stride_bytes;
    }

    {
        // A single b128 A read supplies two four-BF16 K subsets. Lay B out
        // as two 16-row panels matching those subsets: row r maps to
        // (r % 4) + 8 * (r / 4), and the second panel adds four K rows.
        constexpr bool narrow = G::kBlockN == 64;
        const uint32_t row = narrow ? (uint32_t)(tid >> 3)
                                    : (uint32_t)(wid * 4 + (lane >> 4));
        const uint32_t r16 = row & 15u;
        const uint32_t b_k = (r16 & 3u) + 8u * (r16 >> 2) +
                             (narrow ? 4u * (row >> 4) : 0u);
        const uint32_t octet = (uint32_t)(lane & (narrow ? 7 : 15)) ^
                               (4u * (r16 & 1u));
        const uint32_t col = (uint32_t)n_block_start + 8u * octet;
        uint32_t stride = 64u;
        while(stride * 2u <= (uint32_t)S.b_stride_bytes && stride * 2u <= 0x3fffu)
            stride *= 2u;
        const uint32_t offset = b_k * ((uint32_t)S.b_stride_bytes - stride) + col * 2u;
        S.b_pair = (uint64_t)b_k | ((uint64_t)offset << 32);
        S.b_rsrc = make_buffer_rsrc(p.b_ptr, stride,
                                  (uint32_t)((int64_t)p.k * S.b_stride_bytes), false);
    }
    const uint32_t rk = (uint32_t)(lane >> 2);
    const uint32_t ph = 4u * (rk & 1u);
#pragma unroll
    for(int r = 0; r < G::kNTilesPerWave / 2; ++r)
    {
        const uint32_t octet = (uint32_t)(G::kWaveTileN / 8 * wave_n + 4 * r + (lane & 3));
        S.b_rb[r] = (uint32_t)G::kLdsBytesA + rk * (uint32_t)(G::kBlockN * 2) + 16u * (octet ^ ph);
    }

    AOpBank a_bank;
    BOpBank b_bank;

    // Prologue: fill the ring, consume banks 0/1, prime the pre-loop slot.
    // so_a: NN walks K-contiguous A (32 B per stage); TN walks the [K, M]
    // panel row-major (a_stride_bytes per stage) -- see issue_load_and_read.
#pragma unroll
    for(index_t st = 0; st < G::kNumStages; ++st)
    {
        const uint32_t so_a =
            (uint32_t)(st * G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)S.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(st * G::kStageK * S.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            buffer_load_dwordx4_idxen_lds_t(
                S.a_rsrc, S.a_pair, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_idxen_lds(
                S.b_rsrc, S.b_pair, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
            if constexpr(G::kBlockN == 128)
            {
            buffer_load_dwordx4_idxen_lds(
                S.b_rsrc, S.b_pair, so_b + 4u * (uint32_t)S.b_stride_bytes,
                S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage) + 4096u);
            }
        }
        else
        {
            buffer_load_dwordx4_offen_lds(
                S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_idxen_lds(
                S.b_rsrc, S.b_pair, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
            if constexpr(G::kBlockN == 128)
            {
            buffer_load_dwordx4_idxen_lds(
                S.b_rsrc, S.b_pair, so_b + 4u * (uint32_t)S.b_stride_bytes,
                S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage) + 4096u);
            }
        }
    }

    wait_all_buffer_data_arrived<true>(G::kLoadsPerStage * (G::kNumStages - 1));
    if constexpr(kKind == V7UniformKind::NN)
    {
        v7k32::read_a<G, 0>(S.a_rd, a_bank[0]);
    }
    else
    {
        ds_read_m32x16_x4<0>(S.a_rb, a_bank[0]);
    }
    v7k32::read_b<G, 0>(S.b_rb, b_bank[0]);
    wait_all_buffer_data_arrived<true>(G::kLoadsPerStage * (G::kNumStages - 2));
    if constexpr(kKind == V7UniformKind::NN)
    {
        v7k32::read_a<G, G::kLdsBytesPerStage>(S.a_rd, a_bank[1]);
    }
    else
    {
        ds_read_m32x16_x4<G::kLdsBytesPerStage>(S.a_rb, a_bank[1]);
    }
    v7k32::read_b<G, G::kLdsBytesPerStage>(S.b_rb, b_bank[1]);
    wait_all_buffer_data_arrived<true>(G::kLoadsPerStage);
    // lgkmcnt(0), not 8: the pre-loop load below overwrites slot 0, so both
    // prologue bursts must be fully drained first.
    wait_lds_data_arrived<false>(0);
    wait_all_warp_arrived();
    {
        constexpr uint32_t kPreSlot =
            (uint32_t)(((G::kPrefetchDist - 1) % G::kNumStages) * G::kLdsBytesPerStage);
        // Runtime, not constexpr: it wraps to 0 when the K extent is exactly the
        // ring's own depth (K == kNumStages * kStageK), same as the reference.
        const int kPre4 = __builtin_amdgcn_readfirstlane((S.n_stages > (int)G::kPrefetchDist - 1) ? (int)G::kPrefetchDist - 1 : 0);
        const uint32_t so_a =
            (uint32_t)(kPre4 * (int)G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)S.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(kPre4 * (int)G::kStageK * S.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            buffer_load_dwordx4_idxen_lds_t(S.a_rsrc, S.a_pair, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_idxen_lds(S.b_rsrc, S.b_pair, so_b, S.m0_b_base + kPreSlot);
            if constexpr(G::kBlockN == 128)
            {
            buffer_load_dwordx4_idxen_lds(S.b_rsrc, S.b_pair,
                so_b + 4u * (uint32_t)S.b_stride_bytes, S.m0_b_base + kPreSlot + 4096u);
            }
        }
        else
        {
            buffer_load_dwordx4_offen_lds(S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_idxen_lds(S.b_rsrc, S.b_pair, so_b, S.m0_b_base + kPreSlot);
            if constexpr(G::kBlockN == 128)
            {
            buffer_load_dwordx4_idxen_lds(S.b_rsrc, S.b_pair,
                so_b + 4u * (uint32_t)S.b_stride_bytes, S.m0_b_base + kPreSlot + 4096u);
            }
        }
    }

    // The k loop. One divergent branch for the whole kernel (not per step): the
    // exec-mask save/restore is paid once, and each arm is straight-line because
    // kGroupA is a compile-time parameter. The branch must stay DIVERGENT: the
    // uniform-branch variant corrupts on this toolchain (the scheduler reorders
    // the inline-asm waits).
    int kt = 0;
    if(S.wid < (int)G::kWaveGroup)
    {
#pragma unroll 1
        for(bool go = (kt + (int)G::kPhaseCycle <= S.n_stages); go;)
        {
            stage_step<0, true>(S, acc, a_bank, b_bank, kt + 0);
            stage_step<1, true>(S, acc, a_bank, b_bank, kt + 1);
            stage_step<2, true>(S, acc, a_bank, b_bank, kt + 2);
            stage_step<3, true>(S, acc, a_bank, b_bank, kt + 3);
            stage_step<4, true>(S, acc, a_bank, b_bank, kt + 4);
            stage_step<5, true>(S, acc, a_bank, b_bank, kt + 5);
            stage_step<6, true>(S, acc, a_bank, b_bank, kt + 6);
            stage_step<7, true>(S, acc, a_bank, b_bank, kt + 7);
            stage_step<8, true>(S, acc, a_bank, b_bank, kt + 8);
            stage_step<9, true>(S, acc, a_bank, b_bank, kt + 9);
            stage_step<10, true>(S, acc, a_bank, b_bank, kt + 10);
            stage_step<11, true, false>(S, acc, a_bank, b_bank, kt + 11);
            kt += (int)G::kPhaseCycle;
            go = (kt + (int)G::kPhaseCycle <= S.n_stages);
            wait_all_warp_arrived();
        }
        if(kt < S.n_stages) { stage_step<0, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<1, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<2, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<3, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<4, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<5, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<6, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<7, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<8, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<9, true>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<10, true>(S, acc, a_bank, b_bank, kt); ++kt; }
    }
    else
    {
#pragma unroll 1
        for(bool go = (kt + (int)G::kPhaseCycle <= S.n_stages); go;)
        {
            stage_step<0, false>(S, acc, a_bank, b_bank, kt + 0);
            stage_step<1, false>(S, acc, a_bank, b_bank, kt + 1);
            stage_step<2, false>(S, acc, a_bank, b_bank, kt + 2);
            stage_step<3, false>(S, acc, a_bank, b_bank, kt + 3);
            stage_step<4, false>(S, acc, a_bank, b_bank, kt + 4);
            stage_step<5, false>(S, acc, a_bank, b_bank, kt + 5);
            stage_step<6, false>(S, acc, a_bank, b_bank, kt + 6);
            stage_step<7, false>(S, acc, a_bank, b_bank, kt + 7);
            stage_step<8, false>(S, acc, a_bank, b_bank, kt + 8);
            stage_step<9, false>(S, acc, a_bank, b_bank, kt + 9);
            stage_step<10, false>(S, acc, a_bank, b_bank, kt + 10);
            stage_step<11, false, false>(S, acc, a_bank, b_bank, kt + 11);
            kt += (int)G::kPhaseCycle;
            go = (kt + (int)G::kPhaseCycle <= S.n_stages);
            wait_all_warp_arrived();
        }
        if(kt < S.n_stages) { stage_step<0, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<1, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<2, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<3, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<4, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<5, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<6, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<7, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<8, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<9, false>(S, acc, a_bank, b_bank, kt); ++kt; }
        if(kt < S.n_stages) { stage_step<10, false>(S, acc, a_bank, b_bank, kt); ++kt; }
    }
}



} // namespace ck_tile
#pragma clang diagnostic pop
