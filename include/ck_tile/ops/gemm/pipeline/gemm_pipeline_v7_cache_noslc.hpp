// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Experimental A-load cache-policy isolation. CK V7 geometry, mapping, waits,
// operand rings and epilogue remain identical; NN A drops only the SLC modifier.
// Keep this independent so the accepted V7 and its readfirstlane patch stay intact.
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
namespace ck_tile {
template <typename Problem, V7UniformKind kKind>
struct GemmPipelineV7CacheNoSlc
{
    using G       = V7UniformGeometry;
    using AccTile = V7UniformAccTile;

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

    // The LDS the ring occupies (the whole CU budget).
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return G::kLdsBytes; }

    using AOpBank = v7u::vec4_bf16[G::kOpRing][G::kMTilesPerWave];
    using BOpBank = v7u::vec4_bf16[G::kOpRing][G::kNTilesPerWave];

    // Per-thread scalar state: the descriptors / per-lane bases the k loop needs.
    struct Addr
    {
        v7u::buffer_rsrc_t a_rsrc, b_rsrc;
        uint32_t a_vaddr, b_vaddr;      // per-lane global byte offsets (TN, NN-B)
        uint64_t a_pair;                // NN A: {index, offset} idxen pair
        uint32_t a_rd;                  // NN A b64 read base
        uint32_t a_rb[2];               // TN A m32x16 read bases
        uint32_t b_rb[4];               // B m32x16 read bases
        uint32_t m0_a_base, m0_b_base;
        int b_stride_bytes;
        int a_stride_bytes;   // TN: the A panel's row pitch (the k advance);
                              // NN keeps the K-contiguous 2-byte advance and
                              // never reads this.
        int n_stages;
        int wid;
    };

    CK_TILE_DEVICE static void mmac32(AccTile& acc,
                                      const AOpBank& a_bank,
                                      const BOpBank& b_bank,
                                      int bank)
    {
        // The pragma-unrolled loop body is written out for bank 0 so the MMAC
        // burst starts with a `s_setprio 3` as in the reference kernel.
        acc.v[0] = v7u::mmac_bf16(a_bank[bank][0], b_bank[bank][0], acc.v[0]);
        v7u::set_prio<3>();
#pragma unroll
        for(index_t i = 1; i < G::kMmacPerStage; ++i)
        {
            const index_t m = i / G::kNTilesPerWave;
            const index_t n = i % G::kNTilesPerWave;
            acc.v[i]       = v7u::mmac_bf16(a_bank[bank][m], b_bank[bank][n], acc.v[i]);
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
                v7u::ds_read_m32x16_x8<kReadSlot>(s.b_rb, b_bank[kBank]);
                v7u::ds_read_b64_x4<kReadSlot>(
                    s.a_rd, a_bank[kBank][0], a_bank[kBank][1], a_bank[kBank][2], a_bank[kBank][3]);
            }
            else
            {
                v7u::ds_read_m32x16_x8<kReadSlot>(s.b_rb, b_bank[kBank]);
                v7u::ds_read_m32x16_x4<kReadSlot>(s.a_rb, a_bank[kBank]);
            }
        }
        // The k advance rides the soffsets in every form: the descriptor base is
        // never rewritten inside the loop.
        const int kStage   = (kt + G::kPrefetchDist < s.n_stages) ? kt + G::kPrefetchDist : 0;
        // A's k advance: NN walks a K-contiguous [M, K] operand (32 B per
        // 16-wide stage); TN walks the [K, M] panel row-major, so one stage is
        // a_stride_bytes (16 panel rows). The reference kernels advance
        // exactly this way (bf16_gemm_tn_hipc issue_load_and_read).
        const uint32_t so_a =
            (uint32_t)(kStage * G::kStageK *
                       (kKind == V7UniformKind::NN ? 2u : (uint32_t)s.a_stride_bytes));
        const uint32_t so_b = (uint32_t)(kStage * G::kStageK * s.b_stride_bytes);
        if constexpr(kKind == V7UniformKind::NN)
        {
            v7u::buffer_load_dwordx4_idxen_lds(
                s.a_rsrc, s.a_pair, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_offen_lds(
                s.b_rsrc, s.b_vaddr, so_b, s.m0_b_base + kLoadSlot);
        }
        else
        {
            v7u::buffer_load_dwordx4_offen_lds(
                s.a_rsrc, s.a_vaddr, so_a, s.m0_a_base + kLoadSlot);
            v7u::buffer_load_dwordx4_offen_lds(
                s.b_rsrc, s.b_vaddr, so_b, s.m0_b_base + kLoadSlot);
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
            mmac32(acc, a_bank, b_bank, kBankCur);
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(8);
        }
        else
        {
            issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>(s, a_bank, b_bank, kt);
            v7u::wait_lds_data_arrived<false>(8);
            mmac32(acc, a_bank, b_bank, kBankCur);
        }
        v7u::wait_all_buffer_data_arrived<false>(4);
        if constexpr(kBarrier)
        {
            v7u::wait_all_warp_arrived();
        }
    }

    CK_TILE_DEVICE static void run_tile(const V7UniformTileParams& p, AccTile& acc, void* smem);
};

// ---------------------------------------------------------------------------
// The tile body. Written once, with `if constexpr` on the operand layout; the
// instruction order, the waits and the ring arithmetic are the reference
// kernels' (see the header note for what each one is worth).
// ---------------------------------------------------------------------------
template <typename Problem, V7UniformKind kKind>
CK_TILE_DEVICE void GemmPipelineV7CacheNoSlc<Problem, kKind>::run_tile(
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
        const int a_row_global = m_block_start + wid * 32 + (lane >> 1);
        const uint32_t a_kh =
            ((((lane >> 1) & 15) >> 2) & 1) ^ (uint32_t)(lane & 1);
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
            a_row_global < p.m ? (uint32_t)a_row_global : 0xffffffffu;
        S.a_pair = (uint64_t)a_index |
                   ((uint64_t)((uint32_t)(a_row_global * (int)a_off_rem + a_kh * 16)) << 32);
        const uint32_t a_size_bytes = (uint32_t)((int64_t)p.m * a_stride_bytes);
        S.a_rsrc = make_buffer_rsrc(p.a_ptr, a_stride_desc, a_size_bytes, /*swizzle=*/true);
        // Read base: row (lane&15) at +32*row, 8 B k-unit (lane>>4) at +du, with
        // du compensating the swapped rows' [kh1|kh0] placement.
        const uint32_t a_rho = (uint32_t)(lane & 15);
        const uint32_t a_uu  = (uint32_t)(lane >> 4);
        const uint32_t a_t   = (a_rho >> 2) & 1;
        const uint32_t a_du  = a_t ? ((a_uu < 2u) ? 16u + 8u * a_uu : 8u * a_uu - 16u) : 8u * a_uu;
        S.a_rd = (uint32_t)(wave_m_start * 32 + a_rho * 32 + a_du);
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
        // ---- B = [K, N], N-contiguous (both layouts) ----------------------
        const int b_stride_bytes = S.b_stride_bytes;
        // Monotonic walk: lanes 0-31 walk row 2w's full 512 B in address order,
        // lanes 32-63 walk row 2w+1's; the odd row's lane->octet map is XORed by
        // 4 so the two rows land on complementary bank halves. That is the L1
        // request-merge shape (the merged walk measured 310.8 -> 369.1 TF).
        const int b_k_in_stage = wid * 2 + (lane >> 5);
        const int b_n_global =
            n_block_start + 8 * (((lane >> 3) & 3) * 8 + ((lane & 7) ^ (4 * ((lane >> 5) & 1))));
        S.b_vaddr     = (uint32_t)(b_k_in_stage * b_stride_bytes + b_n_global * 2);
        const uint32_t b_size_bytes = (uint32_t)((int64_t)p.k * b_stride_bytes);
        S.b_rsrc      = make_buffer_rsrc(p.b_ptr, (uint32_t)b_stride_bytes, b_size_bytes, false);
    }

    // Read anchors, one per-lane vaddr per read: lane l supplies read r's anchor
    // for octet o = base + 4r + (l&3), row parity rw = (l>>2)&1, window (l>>3):
    //   addr = 1024*(l>>3) + 512*rw + 16*(o ^ (4*rw))
    // (the ^4*rw is the same phase term the walk applies to the odd row, so the
    // two reads hit complementary bank halves). One register per read is what
    // makes "+4 octets" a free immediate; the per-lane anchors are why the read
    // forms take arrays.
    const uint32_t rw = (uint32_t)((lane >> 2) & 1);
    const uint32_t ph = 4u * rw;
    const uint32_t b_rd0 =
        (uint32_t)G::kLdsBytesA + (uint32_t)(1024 * (lane >> 3)) + 512u * rw;
#pragma unroll
    for(int r = 0; r < 4; ++r)
    {
        const uint32_t o = (uint32_t)(16 * wave_n) + 4u * (uint32_t)r + (uint32_t)(lane & 3);
        S.b_rb[r]        = b_rd0 + 16u * (o ^ ph);
    }
    const uint32_t a_rd0 = (uint32_t)(1024 * (lane >> 3)) + 512u * rw;
#pragma unroll
    for(int r = 0; r < 2; ++r)
    {
        const uint32_t o = (uint32_t)(8 * wave_m) + 4u * (uint32_t)r + (uint32_t)(lane & 3);
        S.a_rb[r]        = a_rd0 + 16u * (o ^ ph);
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
            buffer_load_dwordx4_idxen_lds(
                S.a_rsrc, S.a_pair, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_offen_lds(
                S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
        }
        else
        {
            buffer_load_dwordx4_offen_lds(
                S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + (uint32_t)(st * G::kLdsBytesPerStage));
            buffer_load_dwordx4_offen_lds(
                S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + (uint32_t)(st * G::kLdsBytesPerStage));
        }
    }

    wait_all_buffer_data_arrived<true>(2 * (G::kNumStages - 1));
    if constexpr(kKind == V7UniformKind::NN)
    {
        ds_read_b64_x4<0>(S.a_rd, a_bank[0][0], a_bank[0][1], a_bank[0][2], a_bank[0][3]);
    }
    else
    {
        ds_read_m32x16_x4<0>(S.a_rb, a_bank[0]);
    }
    ds_read_m32x16_x8<0>(S.b_rb, b_bank[0]);
    wait_all_buffer_data_arrived<true>(2 * (G::kNumStages - 2));
    if constexpr(kKind == V7UniformKind::NN)
    {
        ds_read_b64_x4<G::kLdsBytesPerStage>(
            S.a_rd, a_bank[1][0], a_bank[1][1], a_bank[1][2], a_bank[1][3]);
    }
    else
    {
        ds_read_m32x16_x4<G::kLdsBytesPerStage>(S.a_rb, a_bank[1]);
    }
    ds_read_m32x16_x8<G::kLdsBytesPerStage>(S.b_rb, b_bank[1]);
    wait_all_buffer_data_arrived<true>(2);
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
            buffer_load_dwordx4_idxen_lds(S.a_rsrc, S.a_pair, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_offen_lds(S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + kPreSlot);
        }
        else
        {
            buffer_load_dwordx4_offen_lds(S.a_rsrc, S.a_vaddr, so_a, S.m0_a_base + kPreSlot);
            buffer_load_dwordx4_offen_lds(S.b_rsrc, S.b_vaddr, so_b, S.m0_b_base + kPreSlot);
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
