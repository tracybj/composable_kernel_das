// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Co., Ltd.
// Adapted from CK BF16 WASP; origin/hash and changes: UPSTREAM_ORIGIN.json.
// INT8 doubles logical K per stage, preserving every physical byte layout.
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"

namespace ck_tile {

struct Int8WaspPriorityAccTile
{
    static constexpr index_t kMIters = 16;
    static constexpr index_t kNIters = 2;

    using acc_vec_t = int32_t __attribute__((ext_vector_type(4)));
    acc_vec_t v[kMIters * kNIters];
};

namespace wasp_int8_priority_detail {

constexpr index_t kVmemSampleStage = 16;

struct WaspVmemSample
{
    uint64_t issue_to_wait_start = 0;
    uint64_t wait_cycles         = 0;
    uint64_t vmcnt4_cycles       = 0;
    uint64_t vmcnt2_cycles       = 0;
    uint32_t valid               = 0;
};

struct WaspVmemDiag
{
    uint64_t issued_at = 0;
    WaspVmemSample sample{};
    bool issued = false;
    bool valid  = false;
};

#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
__device__ static WaspVmemSample g_wasp_vmem_samples[1024 * 16];
#endif

CK_TILE_DEVICE uint64_t wasp_memtime()
{
    return static_cast<uint64_t>(__builtin_amdgcn_s_memtime());
}

using frag_a_t = int32_t __attribute__((ext_vector_type(2)));
using frag_c_t = int32_t __attribute__((ext_vector_type(4)));
using dword4_t = int32_t __attribute__((ext_vector_type(4)));

union alignas(16) stage16_t
{
    dword4_t  d4;
    frag_a_t  frag[2];
};
static_assert(sizeof(stage16_t) == 16, "16 B staging unit");

CK_TILE_DEVICE void ds_write_b128(uint32_t byte_addr, const dword4_t& v)
{
    asm volatile("ds_write_b128 %0, %1" ::"v"(byte_addr), "v"(v) : "memory");
}

CK_TILE_DEVICE frag_c_t mmac_i8(const frag_a_t& a,
                                                  const frag_a_t& b,
                                                  const frag_c_t& c)
{
    frag_c_t out=c;
    asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %0" : "+v"(out) : "v"(a), "v"(b) : "memory");
    return out;
}

CK_TILE_DEVICE dword4_t global_load_b128(const void* p)
{
    return *reinterpret_cast<const dword4_t*>(p);
}

using buffer_rsrc_t = int32_t __attribute__((ext_vector_type(4)));

CK_TILE_DEVICE buffer_rsrc_t wasp_make_buffer_rsrc(const void* ptr,
                                                   uint32_t stride_bytes,
                                                   uint32_t size_bytes,
                                                   bool allow_swizzle = true)
{
    buffer_rsrc_t r;
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    r[0]                = static_cast<int32_t>(addr & 0xffffffffu);
    r[1]                = static_cast<int32_t>(addr >> 32);
    r[2]                = static_cast<int32_t>(size_bytes);
    r[3]                = 0x00020000;

    uint32_t val = (stride_bytes & 0x3fffu) << 16;
#if defined(__gfx936__) || defined(__gfx938__)
    if(allow_swizzle && (stride_bytes % 64u) == 0u)
    {
        const uint32_t s64 = stride_bytes / 64u;
        if((s64 & (s64 - 1u)) == 0u)
        {
            val |= 1u << 30;
        }
    }
#else
    (void)allow_swizzle;
#endif
    r[1] |= static_cast<int32_t>(val);
    return r;
}

CK_TILE_DEVICE void wasp_buffer_load_x8_to_lds(buffer_rsrc_t rsrc,
                                               uint32_t lds_m0,
                                               uint32_t soffset,
                                               uint32_t v0,
                                               uint32_t v1,
                                               uint32_t v2,
                                               uint32_t v3,
                                               uint32_t v4,
                                               uint32_t v5,
                                               uint32_t v6,
                                               uint32_t v7)
{
    asm volatile(
        "s_mov_b32 m0, %1\n\t"
        "buffer_load_dwordx4 %0, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %2, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %3, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %4, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %5, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %6, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %7, %9, %10 offen lds\n\t"
        "s_add_u32 m0, m0, 0x400\n\t"
        "buffer_load_dwordx4 %8, %9, %10 offen lds\n"
        :
        : "v"(v0), "s"(lds_m0), "v"(v1), "v"(v2), "v"(v3), "v"(v4), "v"(v5), "v"(v6),
          "v"(v7), "s"(rsrc), "s"(soffset)
        : "memory");
}

CK_TILE_DEVICE void wasp_buffer_load_b128_x4_idxen_soff2(buffer_rsrc_t rsrc,
                                                         uint64_t p0,
                                                         uint64_t p1,
                                                         uint32_t s_col_base,
                                                         dword4_t& o0,
                                                         dword4_t& o1,
                                                         dword4_t& o2,
                                                         dword4_t& o3)
{
    asm volatile("buffer_load_dwordx4 %0, %4, %6, %7 idxen offen offset:0\n\t"
                 "buffer_load_dwordx4 %1, %4, %6, %7 idxen offen offset:64\n\t"
                 "buffer_load_dwordx4 %2, %5, %6, %7 idxen offen offset:0\n\t"
                 "buffer_load_dwordx4 %3, %5, %6, %7 idxen offen offset:64\n"
                 : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3)
                 : "v"(p0), "v"(p1), "s"(rsrc), "s"(s_col_base)
                 : "memory");
}

CK_TILE_DEVICE dword4_t wasp_buffer_load_b128_idxen(buffer_rsrc_t rsrc,
                                                    uint64_t pair,
                                                    uint32_t soffset)
{
    dword4_t o;
    asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 idxen offen\n"
                 : "=v"(o)
                 : "v"(pair), "s"(rsrc), "s"(soffset)
                 : "memory");
    return o;
}

CK_TILE_DEVICE void wasp_barrier()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_barrier\n");
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void wasp_barrier_maybe()
{
    wasp_barrier();
}

CK_TILE_DEVICE void wasp_sync_lds()
{

    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n" ::: "memory");
}

CK_TILE_DEVICE void wasp_sync_lds_maybe()
{
    wasp_sync_lds();
}

template <int PRIO>
CK_TILE_DEVICE void wasp_set_prio()
{
    static_assert(PRIO >= 0 && PRIO <= 3, "s_setprio operand must be 0..3");
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_setprio %0" ::"n"(PRIO));
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void wasp_wait_vmcnt0()
{
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}

CK_TILE_DEVICE void wasp_wait_vmcnt0(WaspVmemDiag& diag, index_t stage)
{
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
    if(stage == kVmemSampleStage && diag.issued)
    {
        const uint64_t wait_start = wasp_memtime();
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        const uint64_t wait_end = wasp_memtime();
        diag.sample.issue_to_wait_start = wait_start - diag.issued_at;
        diag.sample.wait_cycles         = wait_end - wait_start;
        diag.sample.valid       = 1;
        diag.valid                      = true;
        return;
    }
#else
    (void)diag;
    (void)stage;
#endif
    wasp_wait_vmcnt0();
}

CK_TILE_DEVICE void wasp_wait_vmcnt0_producer(WaspVmemDiag& diag, index_t stage)
{
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
    if(stage == kVmemSampleStage && diag.issued)
    {
        const uint64_t wait_start = wasp_memtime();
        asm volatile("s_waitcnt vmcnt(4)\n" ::: "memory");
        diag.sample.vmcnt4_cycles = wasp_memtime() - wait_start;
        asm volatile("s_waitcnt vmcnt(2)\n" ::: "memory");
        diag.sample.vmcnt2_cycles = wasp_memtime() - wait_start;
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        const uint64_t wait_end = wasp_memtime();
        diag.sample.issue_to_wait_start = wait_start - diag.issued_at;
        diag.sample.wait_cycles         = wait_end - wait_start;
        diag.sample.valid               = 1;
        diag.valid                      = true;
        return;
    }
#else
    (void)diag;
    (void)stage;
#endif
    wasp_wait_vmcnt0();
}

#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_DUMP_A_STAGE)

__device__ static unsigned short g_wasp_a_dump[2 * 256 * 64];

CK_TILE_DEVICE void take_a_stage_snapshot(int slot, const char* lds)
{
    const index_t         lane = ck_tile::get_lane_id();
    const unsigned short* src =
        reinterpret_cast<const unsigned short*>(lds + slot * 32768 + lane * 512);
    for(index_t i = 0; i < 256; ++i)
    {
        g_wasp_a_dump[slot * 256 * 64 + lane * 256 + i] = src[i];
    }
}
#endif

}

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy, index_t ConsumerSplit = 4,index_t AReadBatch = 1,index_t PriorityScope = 0>
struct GemmPipelineInt8WaspPriority : public GemmPipelineAgBgCrCompV3<Problem, Policy>
{
    using Base = GemmPipelineAgBgCrCompV3<Problem, Policy>;

    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;

    static constexpr index_t kNumProducerWaves = 4;
    static constexpr index_t kNumConsumerWaves = 8;

    static constexpr index_t kConsumerSplitWave = ConsumerSplit;

    static constexpr index_t kNumWavesTotal    = kNumProducerWaves + kNumConsumerWaves;

    static constexpr index_t BlockSize = kNumWavesTotal * get_warp_size();

    static_assert(BlockGemmShape::NumWarps == kNumConsumerWaves,
                  "WASP: the tile warp grid must be exactly the consumer waves");
    static_assert(BlockSize == 768, "WASP workgroup is 12 waves of 64");

    static constexpr index_t kStageK        = 128;
    static constexpr index_t kPassK         = 64;
    static constexpr index_t kNumStages     = 2;
    static constexpr index_t kProdSlabRows  = 64;

    static constexpr index_t kWaveN = BlockGemmShape::kN / kNumConsumerWaves;
    static constexpr index_t kProdSlabBytes = kProdSlabRows * kStageK * sizeof(ADataType);
    static constexpr index_t kAStageBytes   = kNumProducerWaves * kProdSlabBytes;
    static constexpr index_t kWaspLdsBytes  = kNumStages * kAStageBytes;

    static_assert(kProdSlabBytes * kNumProducerWaves == kAStageBytes, "A stage budget");
    static_assert(BlockGemmShape::kK == kStageK, "block K must equal the A-stage depth");
    static_assert(BlockGemmShape::kM == kNumProducerWaves * kProdSlabRows,
                  "A rows are split evenly across the producer waves");
    static_assert(BlockGemmShape::kN == kNumConsumerWaves * 32,
                  "each consumer wave owns a 32 wide N stripe");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kWaspLdsBytes; }

    using AccTile = Int8WaspPriorityAccTile;

    static constexpr bool IsWaspPipeline = true;

    struct AProdSrc
    {
        wasp_int8_priority_detail::buffer_rsrc_t rsrc;
        uint32_t                   v_off[8];
    };

    template <typename ADataTypePtr>
    CK_TILE_DEVICE AProdSrc make_a_prod_src(const ADataTypePtr* a_base,
                                            index_t lda,
                                            index_t m_valid,
                                            index_t wid) const
    {
        using namespace wasp_int8_priority_detail;

        const index_t lane = get_lane_id();

        const index_t row_group = lane >> 3;
        const index_t chunk     = lane & 7;
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_REVERSE_PRODUCER_SLAB)

        const index_t slab_id = kNumProducerWaves - 1 - wid;
#else
        const index_t slab_id = wid;
#endif
        const index_t row_base = slab_id * kProdSlabRows;

        const uint32_t pitch_bytes = static_cast<uint32_t>(lda) * sizeof(ADataType);
        const uint32_t size_bytes =
            static_cast<uint32_t>((m_valid > 0 ? m_valid : 0)) * pitch_bytes;

        AProdSrc s;
        s.rsrc = wasp_make_buffer_rsrc(a_base, pitch_bytes, size_bytes, false);

        const uint32_t k_in_panel =
            static_cast<uint32_t>((chunk ^ row_group) * 16);
#pragma unroll
        for(index_t i = 0; i < 8; ++i)
        {
            const index_t  row = row_base + i * 8 + row_group;
            const uint32_t v   = static_cast<uint32_t>(row) * pitch_bytes + k_in_panel;
            s.v_off[i]         = (row < m_valid) ? v : (v | 0x80000000u);
        }

#pragma unroll
        for(index_t i = 0; i < 8; ++i)
        {
            asm volatile("" : "+v"(s.v_off[i]));
        }
        return s;
    }

    CK_TILE_DEVICE void producer_fill(const AProdSrc& src, index_t k_stage, index_t wid) const
    {
        using namespace wasp_int8_priority_detail;

        const index_t slot = (k_stage % kNumStages) * kAStageBytes;

        const uint32_t soffset = static_cast<uint32_t>(k_stage * kStageK) * sizeof(ADataType);
        const uint32_t m0 =
            static_cast<uint32_t>(slot + wid * kProdSlabBytes);

        wasp_buffer_load_x8_to_lds(src.rsrc,
                                   m0,
                                   soffset,
                                   src.v_off[0],
                                   src.v_off[1],
                                   src.v_off[2],
                                   src.v_off[3],
                                   src.v_off[4],
                                   src.v_off[5],
                                   src.v_off[6],
                                   src.v_off[7]);

    }

    using BBuf = wasp_int8_priority_detail::stage16_t[2][Int8WaspPriorityAccTile::kNIters];

    struct AReadBases
    {
        uint32_t h0;
        uint32_t h1;
    };

    struct BPairs
    {
        uint64_t p0;
        uint64_t p1;
    };

    CK_TILE_DEVICE void load_B_stage(wasp_int8_priority_detail::buffer_rsrc_t b_rsrc,
                                     const BPairs& bp,
                                     index_t k_stage,
                                     BBuf& buf,
                                     wasp_int8_priority_detail::WaspVmemDiag& vmem_diag) const
    {
        using namespace wasp_int8_priority_detail;
        const uint32_t s_col_base = static_cast<uint32_t>(k_stage * kStageK) * sizeof(ADataType);

        wasp_buffer_load_b128_x4_idxen_soff2(b_rsrc, bp.p0, bp.p1, s_col_base,
                                             buf[0][0].d4,
                                             buf[1][0].d4,
                                             buf[0][1].d4,
                                             buf[1][1].d4);
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
        if(k_stage == wasp_int8_priority_detail::kVmemSampleStage)
        {
            vmem_diag.issued_at = wasp_int8_priority_detail::wasp_memtime();
            vmem_diag.issued = true;
        }
#else
        (void)vmem_diag;
#endif
    }

    static constexpr uint32_t kBHalfBytes = 64;

    template <index_t SLOT>
    CK_TILE_DEVICE static void
    ds_read_b128_x8_imm(uint32_t base,
                        wasp_int8_priority_detail::dword4_t& o0,
                        wasp_int8_priority_detail::dword4_t& o1,
                        wasp_int8_priority_detail::dword4_t& o2,
                        wasp_int8_priority_detail::dword4_t& o3,
                        wasp_int8_priority_detail::dword4_t& o4,
                        wasp_int8_priority_detail::dword4_t& o5,
                        wasp_int8_priority_detail::dword4_t& o6,
                        wasp_int8_priority_detail::dword4_t& o7)
    {
        static_assert(SLOT + 14336 <= 65535,
                      "ds_read_b128 imm offset overflows the 16-bit unsigned field");
        asm volatile("ds_read_b128 %0, %8 offset:%c9\n\t"
                     "ds_read_b128 %1, %8 offset:%c10\n\t"
                     "ds_read_b128 %2, %8 offset:%c11\n\t"
                     "ds_read_b128 %3, %8 offset:%c12\n\t"
                     "ds_read_b128 %4, %8 offset:%c13\n\t"
                     "ds_read_b128 %5, %8 offset:%c14\n\t"
                     "ds_read_b128 %6, %8 offset:%c15\n\t"
                     "ds_read_b128 %7, %8 offset:%c16\n"
                     : "=&v"(o0), "=&v"(o1), "=&v"(o2), "=&v"(o3),
                       "=&v"(o4), "=&v"(o5), "=&v"(o6), "=&v"(o7)
                     : "v"(base),
                       "n"(SLOT + 0),     "n"(SLOT + 2048),  "n"(SLOT + 4096),
                       "n"(SLOT + 6144),  "n"(SLOT + 8192),  "n"(SLOT + 10240),
                       "n"(SLOT + 12288), "n"(SLOT + 14336)
                     : "memory");
    }

    template <index_t SLOT, bool H1>
    CK_TILE_DEVICE void a_read_half_low(
        const AReadBases& b,
        wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters]) const
    {
        ds_read_b128_x8_imm<SLOT>(H1 ? b.h1 : b.h0,
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[0]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[1]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[2]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[3]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[4]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[5]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[6]),
                                  reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[7]));
    }

    template <index_t SLOT, bool H1>
    CK_TILE_DEVICE void a_read_half_high(
        const AReadBases& b,
        wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters]) const
    {
        ds_read_b128_x8_imm<SLOT + 16384>(H1 ? b.h1 : b.h0,
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[8]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[9]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[10]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[11]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[12]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[13]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[14]),
                                          reinterpret_cast<wasp_int8_priority_detail::dword4_t&>(a_regs[15]));
    }

    CK_TILE_DEVICE void mmac_half(const wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                                  const wasp_int8_priority_detail::stage16_t (&b_half)[Int8WaspPriorityAccTile::kNIters],
                                  AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;

        mmac_one<0, 0, 0>(a_regs, b_half, acc);
        wasp_set_prio<3>();

        static_for<1, Int8WaspPriorityAccTile::kNIters, 1>{}([&](auto n) {
            mmac_one<0, 0, n.value>(a_regs, b_half, acc);
        });
        static_for<1, Int8WaspPriorityAccTile::kMIters, 1>{}([&](auto m) {
            static_for<0, Int8WaspPriorityAccTile::kNIters, 1>{}([&](auto n) {
                mmac_one<0, m.value, n.value>(a_regs, b_half, acc);
            });
        });
        static_for<0, Int8WaspPriorityAccTile::kMIters, 1>{}([&](auto m) {
            static_for<0, Int8WaspPriorityAccTile::kNIters, 1>{}([&](auto n) {
                mmac_one<1, m.value, n.value>(a_regs, b_half, acc);
            });
        });
        wasp_set_prio<0>();
    }

    template <index_t KK, index_t M, index_t N>
    CK_TILE_DEVICE void mmac_one(const wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                                 const wasp_int8_priority_detail::stage16_t (&b_half)[Int8WaspPriorityAccTile::kNIters],
                                 AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;
        constexpr index_t slot = M * Int8WaspPriorityAccTile::kNIters + N;
        acc.v[slot] = mmac_i8(a_regs[M].frag[KK], b_half[N].frag[KK], acc.v[slot]);
    }

    static_assert(AReadBatch==1 || AReadBatch==2 || AReadBatch==4 || AReadBatch==8 || AReadBatch==16);
    template<index_t SLOT,bool H1> CK_TILE_DEVICE void consume_short(const AReadBases& rb,const wasp_int8_priority_detail::stage16_t (&b)[2],AccTile& acc) const {
        using namespace wasp_int8_priority_detail;
        static_for<0,16/AReadBatch,1>{}([&](auto gg){constexpr int base_m=decltype(gg)::value*AReadBatch;
            stage16_t a[AReadBatch];const uint32_t base=H1?rb.h1:rb.h0;
            static_for<0,AReadBatch,1>{}([&](auto rr){constexpr int i=decltype(rr)::value;
                asm volatile("ds_read_b128 %0, %1 offset:%c2" : "=&v"(a[i].d4) : "v"(base),"n"(SLOT+(base_m+i)*2048) : "memory");
            });
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            static_for<0,2,1>{}([&](auto kk){constexpr int k=decltype(kk)::value;
                static_for<0,AReadBatch,1>{}([&](auto rr){constexpr int i=decltype(rr)::value;
                    static_for<0,2,1>{}([&](auto nn){constexpr int n=decltype(nn)::value;
                        acc.v[(base_m+i)*2+n]=mmac_i8(a[i].frag[k],b[n].frag[k],acc.v[(base_m+i)*2+n]);
                        if constexpr(k==0 && i==0 && n==0 && (PriorityScope==1 || (PriorityScope==2 && base_m==0)))wasp_set_prio<3>();
                    });
                });
            });
            if constexpr(PriorityScope==1)wasp_set_prio<0>();
        });
        if constexpr(PriorityScope==2)wasp_set_prio<0>();
    }

    template <index_t CUR>
    CK_TILE_DEVICE void c_I1(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             const AReadBases& a_rb,
                             wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                             wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                             wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;
        (void)consumer_id;
        (void)num_loop;
        constexpr index_t SLOT = CUR * kAStageBytes;

        wasp_int8_priority_detail::wasp_wait_vmcnt0(vmem_diag, k_stage);
        consume_short<SLOT,false>(a_rb,b_regs[CUR][0],acc);
    }

    template <index_t CUR>
    CK_TILE_DEVICE void c_I2(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             const AReadBases& a_rb,
                             wasp_int8_priority_detail::buffer_rsrc_t b_rsrc,
                             const BPairs& b_pairs,
                             wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                             wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                             wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)consumer_id;

        if(k_stage+1<num_loop)load_B_stage(b_rsrc,b_pairs,k_stage+1,b_regs[CUR^1],vmem_diag);
        consume_short<SLOT,true>(a_rb,b_regs[CUR][1],acc);
    }

    template <index_t CUR, bool IS_FIRST>
    CK_TILE_DEVICE void cp_I1(index_t consumer_id,
                              index_t k_stage,
                              index_t num_loop,
                              const AReadBases& a_rb,
                              wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                              wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                              wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                              AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;
        (void)consumer_id;
        (void)num_loop;
        constexpr index_t SLOT = CUR * kAStageBytes;
        if constexpr(!IS_FIRST)
        {

            mmac_half(a_regs, b_regs[CUR ^ 1][1], acc);
        }

        wasp_int8_priority_detail::wasp_wait_vmcnt0(vmem_diag, k_stage);
        a_read_half_low<SLOT, false>(a_rb, a_regs);
        a_read_half_high<SLOT, false>(a_rb, a_regs);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }

    template <index_t CUR>
    CK_TILE_DEVICE void cp_I2(index_t consumer_id,
                              index_t k_stage,
                              index_t num_loop,
                              const AReadBases& a_rb,
                              wasp_int8_priority_detail::buffer_rsrc_t b_rsrc,
                              const BPairs& b_pairs,
                              wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                              wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                              wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                              AccTile& acc) const
    {
        using namespace wasp_int8_priority_detail;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)consumer_id;

        mmac_half(a_regs, b_regs[CUR][0], acc);

        a_read_half_low<SLOT, true>(a_rb, a_regs);
        if(k_stage + 1 < num_loop)
        {
            load_B_stage(b_rsrc, b_pairs, k_stage + 1, b_regs[CUR ^ 1], vmem_diag);
        }
        a_read_half_high<SLOT, true>(a_rb, a_regs);
        asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
    }

    CK_TILE_DEVICE void consumer_run_c(index_t consumer_id,
                                                 index_t num_loop,
                                                 const AReadBases& a_rb,
                                                 wasp_int8_priority_detail::buffer_rsrc_t b_rsrc,
                                                 const BPairs& b_pairs,
                                                 wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                                                 wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                                                 wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                                                 AccTile& acc) const
    {
        for(index_t k = 0; k < num_loop; k += 2)
        {
            c_I1<0>(consumer_id, k, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            c_I2<0>(consumer_id,
                    k,
                    num_loop,
                    a_rb,
                    b_rsrc,
                    b_pairs,
                    b_regs,
                    a_regs,
                    vmem_diag,
                    acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            if(k + 1 < num_loop)
            {
                c_I1<1>(
                    consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                c_I2<1>(consumer_id,
                        k + 1,
                        num_loop,
                        a_rb,
                        b_rsrc,
                        b_pairs,
                        b_regs,
                        a_regs,
                        vmem_diag,
                        acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
            }
        }
    }

    CK_TILE_DEVICE void consumer_run_cp(index_t consumer_id,
                                                   index_t num_loop,
                                                   const AReadBases& a_rb,
                                                   wasp_int8_priority_detail::buffer_rsrc_t b_rsrc,
                                                   const BPairs& b_pairs,
                                                   wasp_int8_priority_detail::stage16_t (&b_regs)[2][2][Int8WaspPriorityAccTile::kNIters],
                                                   wasp_int8_priority_detail::stage16_t (&a_regs)[Int8WaspPriorityAccTile::kMIters],
                                                   wasp_int8_priority_detail::WaspVmemDiag& vmem_diag,
                                                   AccTile& acc) const
    {
        if(num_loop == 0)
        {
            return;
        }
        if((num_loop & 1) != 0)
        {
            cp_I1<0, true>(
                consumer_id, 0, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            cp_I2<0>(
                consumer_id, 0, num_loop, a_rb, b_rsrc, b_pairs, b_regs, a_regs, vmem_diag, acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            for(index_t k = 1; k < num_loop; k += 2)
            {
                cp_I1<1, false>(
                    consumer_id, k, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I2<1>(consumer_id,
                         k,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I1<0, false>(
                    consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I2<0>(consumer_id,
                         k + 1,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
            }

            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            mmac_half(a_regs, b_regs[0][1], acc);
        }
        else
        {
            cp_I1<0, true>(
                consumer_id, 0, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            cp_I2<0>(
                consumer_id, 0, num_loop, a_rb, b_rsrc, b_pairs, b_regs, a_regs, vmem_diag, acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();

            for(index_t k = 1; k < num_loop - 1; k += 2)
            {
                cp_I1<1, false>(
                    consumer_id, k, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I2<1>(consumer_id,
                         k,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I1<0, false>(
                    consumer_id, k + 1, num_loop, a_rb, b_regs, a_regs, vmem_diag, acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
                cp_I2<0>(consumer_id,
                         k + 1,
                         num_loop,
                         a_rb,
                         b_rsrc,
                         b_pairs,
                         b_regs,
                         a_regs,
                         vmem_diag,
                         acc);
                wasp_int8_priority_detail::wasp_barrier_maybe();
            }
            cp_I1<1, false>(consumer_id,
                            num_loop - 1,
                            num_loop,
                            a_rb,
                            b_regs,
                            a_regs,
                            vmem_diag,
                            acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            cp_I2<1>(consumer_id,
                     num_loop - 1,
                     num_loop,
                     a_rb,
                     b_rsrc,
                     b_pairs,
                     b_regs,
                     a_regs,
                     vmem_diag,
                     acc);
            wasp_int8_priority_detail::wasp_barrier_maybe();

            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            mmac_half(a_regs, b_regs[1][1], acc);
        }
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE AccTile operator()(const ADramBlockWindow& a_dram_block_window,
                                      const BDramBlockWindow& b_dram_block_window,
                                      index_t num_loop,
                                      bool ,
                                      TailNumber ,
                                      void* p_smem) const
    {

        const auto  a_view = a_dram_block_window.get_bottom_tensor_view();
        const auto& a_desc = a_view.get_tensor_descriptor();
        const auto  a_org  = a_dram_block_window.get_window_origin();
        const auto* a_base = a_view.get_buffer_view().p_data_ +
                             a_desc.calculate_offset(make_multi_index(a_org[0], a_org[1]));
        const index_t lda = a_desc.calculate_offset(make_multi_index(1, 0)) -
                            a_desc.calculate_offset(make_multi_index(0, 0));

        const index_t a_m     = a_desc.get_lengths()[number<0>{}];
        const index_t m_valid = a_m - a_org[0];

        const auto  b_view = b_dram_block_window.get_bottom_tensor_view();
        const auto& b_desc = b_view.get_tensor_descriptor();
        const auto  b_org  = b_dram_block_window.get_window_origin();
        const auto* b_base = b_view.get_buffer_view().p_data_ +
                             b_desc.calculate_offset(make_multi_index(b_org[0], b_org[1]));

        const index_t b_n_stride = b_desc.calculate_offset(make_multi_index(1, 0)) -
                                   b_desc.calculate_offset(make_multi_index(0, 0));
        const index_t b_n_len = b_desc.get_lengths()[number<0>{}];

        return run_raw(a_base,
                       lda,
                       m_valid,
                       b_base,
                       b_n_stride,
                       b_n_len,
                       static_cast<index_t>(b_org[0]),
                       num_loop,
                       p_smem);
    }

    CK_TILE_DEVICE AccTile run_raw(const ADataType* a_base,
                                   index_t lda,
                                   index_t m_valid,
                                   const BDataType* b_base,
                                   index_t b_n_stride,
                                   index_t b_n_len,
                                   index_t b_n_origin,
                                   index_t num_loop,
                                   void* p_smem) const
    {

        AccTile acc{};
        run_raw_into(a_base,
                     lda,
                     m_valid,
                     b_base,
                     b_n_stride,
                     b_n_len,
                     b_n_origin,
                     num_loop,
                     acc,
                     p_smem);
        return acc;
    }

    CK_TILE_DEVICE void run_raw_into(const ADataType* a_base,
                                     index_t lda,
                                     index_t m_valid,
                                     const BDataType* b_base,
                                     index_t b_n_stride,
                                     index_t b_n_len,
                                     index_t b_n_origin,
                                     index_t num_loop,
                                     AccTile& acc,
                                     void* p_smem) const
    {
        char* const lds   = static_cast<char*>(p_smem);
        const index_t wid = get_warp_id();

        wasp_int8_priority_detail::WaspVmemDiag vmem_diag{};

        const uint32_t b_pitch_bytes = static_cast<uint32_t>(b_n_stride) * sizeof(ADataType);
        const uint32_t b_size_bytes =
            static_cast<uint32_t>(b_n_len - b_n_origin) * b_pitch_bytes;
        const wasp_int8_priority_detail::buffer_rsrc_t b_rsrc =
            wasp_int8_priority_detail::wasp_make_buffer_rsrc(b_base, b_pitch_bytes, b_size_bytes,
                                               true);

        wasp_int8_priority_detail::wasp_sync_lds_maybe();

        if(wid < kNumProducerWaves)
        {

            const auto a_src = make_a_prod_src(a_base, lda, m_valid, wid);

            producer_fill(a_src, 0, wid);
            wasp_int8_priority_detail::wasp_barrier_maybe();
            wasp_int8_priority_detail::wasp_wait_vmcnt0();
            wasp_int8_priority_detail::wasp_barrier_maybe();

            for(index_t k = 0; k + 1 < num_loop; ++k)
            {

                producer_fill(a_src, k + 1, wid);
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
                if(k + 1 == wasp_int8_priority_detail::kVmemSampleStage)
                {
                    vmem_diag.issued_at = wasp_int8_priority_detail::wasp_memtime();
                    vmem_diag.issued = true;
                }
#endif
                wasp_int8_priority_detail::wasp_barrier_maybe();
                wasp_int8_priority_detail::wasp_wait_vmcnt0_producer(vmem_diag, k + 1);
                wasp_int8_priority_detail::wasp_barrier_maybe();
            }

            if(num_loop > 0)
            {
                wasp_int8_priority_detail::wasp_barrier_maybe();
                wasp_int8_priority_detail::wasp_wait_vmcnt0_producer(vmem_diag, num_loop);
                wasp_int8_priority_detail::wasp_barrier_maybe();
            }
        }
        else
        {

            acc = AccTile{};
            const index_t consumer_id = wid - kNumProducerWaves;

            wasp_int8_priority_detail::stage16_t a_regs[Int8WaspPriorityAccTile::kMIters];

            const uint32_t lane    = static_cast<uint32_t>(get_lane_id());
            const uint32_t row     = lane & 15u;
            const uint32_t r7      = row & 7u;
            const uint32_t colg    = lane >> 4;
            const uint32_t rowbase = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                                     row * (kStageK * sizeof(ADataType));
            const AReadBases a_rb{rowbase + ((colg ^ r7) * 16u),
                                  rowbase + (((colg + 4u) ^ r7) * 16u)};
            const uint32_t nrow  = static_cast<uint32_t>(consumer_id) * 32u + row;
            const uint32_t inrow = colg * 16u;
            const BPairs b_pr{(static_cast<uint64_t>(inrow) << 32) | nrow,
                              (static_cast<uint64_t>(inrow) << 32) | (nrow + 16u)};

            wasp_int8_priority_detail::stage16_t b_regs[2][2][Int8WaspPriorityAccTile::kNIters];

            if(num_loop > 0)
            {
                load_B_stage(b_rsrc, b_pr, 0, b_regs[0], vmem_diag);
            }
            wasp_int8_priority_detail::wasp_barrier_maybe();
            wasp_int8_priority_detail::wasp_barrier_maybe();

            if constexpr(ConsumerSplit >= kNumConsumerWaves)
            {
                consumer_run_c(consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
            else if constexpr(ConsumerSplit == 0)
            {
                consumer_run_cp(consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
            else if(consumer_id < kConsumerSplitWave)
            {
                consumer_run_c(
                    consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
            else
            {
                consumer_run_cp(
                    consumer_id, num_loop, a_rb, b_rsrc, b_pr, b_regs, a_regs, vmem_diag, acc);
            }
        }

        wasp_int8_priority_detail::wasp_sync_lds_maybe();

#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
        if(get_lane_id() == 0)
        {
            const index_t cta = get_block_id();
            if(cta < 1024 && wid < 16)
            {
                wasp_int8_priority_detail::g_wasp_vmem_samples[cta * 16 + wid] = vmem_diag.sample;
            }
        }
#endif

    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE AccTile operator()(const ADramBlockWindow& a_dram_block_window,
                                      const BDramBlockWindow& b_dram_block_window,
                                      index_t num_loop,
                                      void* p_smem) const
    {
        return (*this)(a_dram_block_window,
                       b_dram_block_window,
                       num_loop,
                       true,
                       TailNumber::Full,
                       p_smem);
    }
};

}
