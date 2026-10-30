// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Co., Ltd.
// Adapted from CK BF16 WASP; origin and adaptation: docs/grouped_int8_nt_wasp_origin.json.
// INT8 doubles logical K per stage, preserving every physical byte layout.
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"

namespace ck_tile {

struct Int8WaspM192RotatedAccTile
{
    static constexpr index_t kMIters = 12;
    static constexpr index_t kNIters = 2;

    using acc_vec_t = int32_t __attribute__((ext_vector_type(4)));
    acc_vec_t v[kMIters * kNIters];
};

namespace grouped_int8_nt_wasp_m192_rotated_detail {

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

CK_TILE_DEVICE void wasp_buffer_load_x6_to_lds(buffer_rsrc_t rsrc,uint32_t base,
        uint32_t soffset,const uint32_t (&offsets)[6])
{
    #pragma unroll
    for(int i=0;i<6;++i){
        const uint32_t dest=base+i*4096u;
        asm volatile("s_mov_b32 m0, %1\n\t"
                     "buffer_load_dwordx4 %0, %2, %3 offen lds\n"
                     :: "v"(offsets[i]),"s"(dest),"s"(rsrc),"s"(soffset) : "memory","m0");
    }
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

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy, index_t ConsumerSplit = 8,index_t AReadBatch = 1,bool APrefetch = false,bool OneBarrier = false,index_t RecordCap = 0,bool EarlyB = false,bool PackedB = false,index_t BAddress = 0,index_t MmacPriority = 1>
struct GemmPipelineInt8WaspM192Rotated : public GemmPipelineAgBgCrCompV3<Problem, Policy>
{
    using Base = GemmPipelineAgBgCrCompV3<Problem, Policy>;

    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;

    static constexpr index_t kNumProducerWaves = 4;
    static constexpr index_t kNumConsumerWaves = BlockGemmShape::kN / 32;

    static constexpr index_t kConsumerSplitWave = ConsumerSplit;

    static constexpr index_t kNumWavesTotal    = kNumProducerWaves + kNumConsumerWaves;

    static constexpr index_t BlockSize = kNumWavesTotal * get_warp_size();

    static_assert(BlockGemmShape::NumWarps == kNumConsumerWaves,
                  "WASP: the tile warp grid must be exactly the consumer waves");
    static_assert(BlockSize == (4 + kNumConsumerWaves) * 64, "WASP workgroup is four producer waves and N/32 consumers");

    static constexpr index_t kStageK        = 128;
    static constexpr index_t kPassK         = 64;
    static constexpr index_t kNumStages     = 2;
    static constexpr index_t kProdSlabRows  = 48;

    static constexpr index_t kWaveN = BlockGemmShape::kN / kNumConsumerWaves;
    static constexpr index_t kProdSlabBytes = kProdSlabRows * kStageK * sizeof(ADataType);
    static constexpr index_t kAStageBytes   = 32768;
    static constexpr index_t kWaspLdsBytes  = kNumStages * kAStageBytes;

    static_assert(kProdSlabBytes * kNumProducerWaves <= kAStageBytes, "A stage budget");
    static_assert(BlockGemmShape::kK == kStageK, "block K must equal the A-stage depth");
    static_assert(BlockGemmShape::kM == kNumProducerWaves * kProdSlabRows,
                  "A rows are split evenly across the producer waves");
    static_assert(BlockGemmShape::kN == kNumConsumerWaves * 32,
                  "each consumer wave owns a 32 wide N stripe");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kWaspLdsBytes; }

    using AccTile = Int8WaspM192RotatedAccTile;

    static constexpr bool IsWaspPipeline = true;

    struct AProdSrc
    {
        grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t rsrc;
        uint32_t                   v_off[6];
    };

    template <typename ADataTypePtr>
    CK_TILE_DEVICE AProdSrc make_a_prod_src(const ADataTypePtr* a_base,
                                            index_t lda,
                                            index_t m_valid,
                                            index_t wid) const
    {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;

        const index_t lane = get_lane_id();

        const index_t row_group = lane >> 3;
        const index_t chunk     = lane & 7;
        const uint32_t pitch_bytes = static_cast<uint32_t>(lda) * sizeof(ADataType);
        const uint32_t size_bytes =
            static_cast<uint32_t>((m_valid > 0 ? m_valid : 0)) * pitch_bytes;

        AProdSrc s;
        s.rsrc = wasp_make_buffer_rsrc(a_base, pitch_bytes, size_bytes, false);

        const uint32_t k_in_panel =
            static_cast<uint32_t>(chunk * 16);
#pragma unroll
        for(index_t i = 0; i < 6; ++i)
        {
            const index_t  row = wid + 4 * row_group + 32 * i;
            const uint32_t v   = static_cast<uint32_t>(row) * pitch_bytes + k_in_panel;
            s.v_off[i]         = (row < m_valid) ? v : (v | 0x80000000u);
        }

#pragma unroll
        for(index_t i = 0; i < 6; ++i)
        {
            asm volatile("" : "+v"(s.v_off[i]));
        }
        return s;
    }
    CK_TILE_DEVICE void producer_fill(const AProdSrc& src, index_t k_stage, index_t wid) const
    {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;

        const index_t slot = (k_stage % kNumStages) * kAStageBytes;

        const uint32_t soffset = static_cast<uint32_t>(k_stage * kStageK) * sizeof(ADataType);
        const uint32_t m0 =
            __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(slot + wid * 1024u + (wid << 17)));

        wasp_buffer_load_x6_to_lds(src.rsrc,m0,soffset,src.v_off);

    }
    using BBuf = grouped_int8_nt_wasp_m192_rotated_detail::stage16_t[2][Int8WaspM192RotatedAccTile::kNIters];
    struct AReadBases { uint32_t h0e,h0o,h1e,h1o; };
    struct BPairs { uint64_t p0,p1; uint32_t half_stride=0; };
    CK_TILE_DEVICE void load_B_stage(grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t b_rsrc,
                                     const BPairs& bp,
                                     index_t k_stage,
                                     BBuf& buf,
                                     grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag& vmem_diag) const
    {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;
        if constexpr(PackedB)
        {
            // Physical [K/64,N/16,4,16,16]; every N16/K64 fragment is contiguous.
            const uint32_t lo = static_cast<uint32_t>(bp.p0), hi = static_cast<uint32_t>(bp.p1);
            const uint32_t s0 = static_cast<uint32_t>(k_stage) * 2u * bp.half_stride;
            const uint32_t s1 = s0 + bp.half_stride;
            asm volatile("buffer_load_dwordx4 %0, %4, %6, %7 offen\n\t"
                         "buffer_load_dwordx4 %1, %4, %6, %8 offen\n\t"
                         "buffer_load_dwordx4 %2, %5, %6, %7 offen\n\t"
                         "buffer_load_dwordx4 %3, %5, %6, %8 offen\n"
                : "=&v"(buf[0][0].d4),"=&v"(buf[1][0].d4),"=&v"(buf[0][1].d4),"=&v"(buf[1][1].d4)
                : "v"(lo),"v"(hi),"s"(b_rsrc),"s"(s0),"s"(s1) : "memory");
            (void)vmem_diag;
            return;
        }
        const uint32_t s_col_base = static_cast<uint32_t>(k_stage * kStageK) * sizeof(ADataType);
        if constexpr(BAddress != 0)
        {
            const uint32_t p0 = static_cast<uint32_t>(bp.p0),p1 = static_cast<uint32_t>(bp.p1);
            asm volatile("buffer_load_dwordx4 %0, %4, %6, %7 offen offset:0\n\t"
                         "buffer_load_dwordx4 %1, %4, %6, %7 offen offset:64\n\t"
                         "buffer_load_dwordx4 %2, %5, %6, %7 offen offset:0\n\t"
                         "buffer_load_dwordx4 %3, %5, %6, %7 offen offset:64\n"
                : "=&v"(buf[0][0].d4),"=&v"(buf[1][0].d4),"=&v"(buf[0][1].d4),"=&v"(buf[1][1].d4)
                : "v"(p0),"v"(p1),"s"(b_rsrc),"s"(s_col_base) : "memory");
            (void)vmem_diag;
            return;
        }
        wasp_buffer_load_b128_x4_idxen_soff2(b_rsrc, bp.p0, bp.p1, s_col_base,
                                             buf[0][0].d4,
                                             buf[1][0].d4,
                                             buf[0][1].d4,
                                             buf[1][1].d4);
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
        if(k_stage == grouped_int8_nt_wasp_m192_rotated_detail::kVmemSampleStage)
        {
            vmem_diag.issued_at = grouped_int8_nt_wasp_m192_rotated_detail::wasp_memtime();
            vmem_diag.issued = true;
        }
#else
        (void)vmem_diag;
#endif
    }
    template<int OFF> CK_TILE_DEVICE void read_prefetch(uint32_t addr,grouped_int8_nt_wasp_m192_rotated_detail::dword4_t& out) const {
        asm volatile("ds_read_b128 %0, %1 offset:%c2" : "=&v"(out) : "v"(addr),"n"(OFF) : "memory");
    }
    static_assert(ConsumerSplit==8 || (ConsumerSplit==4 && !OneBarrier));
    static_assert(AReadBatch==12 && !APrefetch);
    static_assert(MmacPriority==0 || MmacPriority==1);
    using Stage=grouped_int8_nt_wasp_m192_rotated_detail::stage16_t;
    using Diag=grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag;
    using Buffer=grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t;
    using BRegs=Stage[2][2][2];using ARegs=Stage[12];
    template<int SLOT,bool High,int MI>
    CK_TILE_DEVICE void read_a(const AReadBases& rb,Stage& a) const {
        const uint32_t base=High?(MI%2?rb.h1o:rb.h1e):(MI%2?rb.h0o:rb.h0e);
        read_prefetch<SLOT+(MI/2)*4096>(base,a.d4);
    }
    CK_TILE_DEVICE void mmac_full(const ARegs& a,const Stage (&b)[2],AccTile& acc) const {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;
        if constexpr(MmacPriority)wasp_set_prio<MmacPriority>();
        static_for<0,2,1>{}([&](auto kk){constexpr int k=decltype(kk)::value;
            static_for<0,12,1>{}([&](auto mm){constexpr int m=decltype(mm)::value;
                static_for<0,2,1>{}([&](auto nn){constexpr int n=decltype(nn)::value;
                    acc.v[m*2+n]=mmac_i8(a[m].frag[k],b[n].frag[k],acc.v[m*2+n]);
                });
            });
        });
        if constexpr(MmacPriority)wasp_set_prio<0>();
    }
    template<int SLOT,bool High>
    CK_TILE_DEVICE void load_a_half(const AReadBases& rb,ARegs& a) const {
        static_for<0,12,1>{}([&](auto mm){constexpr int m=decltype(mm)::value;read_a<SLOT,High,m>(rb,a[m]);});
    }
    template<index_t SLOT,bool H1>
    CK_TILE_DEVICE void consume_short(const AReadBases& rb,const Stage (&b)[2],AccTile& acc) const {
        ARegs a;load_a_half<SLOT,H1>(rb,a);
        __builtin_amdgcn_s_waitcnt(0xc07f); // All LDS reads,leave independent VMEM outstanding.
        mmac_full(a,b,acc);
    }
    template<int CUR,bool First>
    CK_TILE_DEVICE void cp_stage(index_t k,index_t loops,const AReadBases& rb,Buffer br,
        const BPairs& bp,BRegs& b,ARegs& a,Diag& diag,AccTile& acc) const {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;
        if constexpr(!First)mmac_full(a,b[CUR^1][1],acc);
        __builtin_amdgcn_s_waitcnt(0x0f70);
        load_a_half<CUR*kAStageBytes,false>(rb,a);
        __builtin_amdgcn_s_waitcnt(0xc07f);
        wasp_barrier_maybe();
        mmac_full(a,b[CUR][0],acc);
        static_for<0,6,1>{}([&](auto mm){constexpr int m=decltype(mm)::value;read_a<CUR*kAStageBytes,true,m>(rb,a[m]);});
        if(k+1<loops)load_B_stage(br,bp,k+1,b[CUR^1],diag);
        static_for<6,12,1>{}([&](auto mm){constexpr int m=decltype(mm)::value;read_a<CUR*kAStageBytes,true,m>(rb,a[m]);});
        __builtin_amdgcn_s_waitcnt(0xc07f);
        wasp_barrier_maybe(); // High A iscaptured beforeits LDS slot canbereused.
    }
    CK_TILE_DEVICE void consumer_run_cp(index_t,index_t loops,const AReadBases& rb,Buffer br,
        const BPairs& bp,BRegs& b,ARegs& a,Diag& diag,AccTile& acc) const {
        if(loops==0)return;
        cp_stage<0,true>(0,loops,rb,br,bp,b,a,diag,acc);
        if(loops&1){
            for(index_t k=1;k<loops;k+=2){
                cp_stage<1,false>(k,loops,rb,br,bp,b,a,diag,acc);
                cp_stage<0,false>(k+1,loops,rb,br,bp,b,a,diag,acc);
            }
            mmac_full(a,b[0][1],acc);
        }else{
            for(index_t k=1;k<loops-1;k+=2){
                cp_stage<1,false>(k,loops,rb,br,bp,b,a,diag,acc);
                cp_stage<0,false>(k+1,loops,rb,br,bp,b,a,diag,acc);
            }
            cp_stage<1,false>(loops-1,loops,rb,br,bp,b,a,diag,acc);
            mmac_full(a,b[1][1],acc);
        }
    }
    template <index_t CUR>
    CK_TILE_DEVICE void c_I1(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t b_rsrc,
                             const BPairs& b_pairs,
                             const AReadBases& a_rb,
                             grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&b_regs)[2][2][Int8WaspM192RotatedAccTile::kNIters],
                             grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&a_regs)[Int8WaspM192RotatedAccTile::kMIters],
                             grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;
        (void)consumer_id;
        (void)num_loop;
        constexpr index_t SLOT = CUR * kAStageBytes;

        __builtin_amdgcn_s_waitcnt(0x0f70);
        // The opposite B register slot is free after the previous stage. Issue
        // the next read before low+high MMAC instead of before high alone.
        if constexpr(EarlyB)
            if(k_stage+1<num_loop)load_B_stage(b_rsrc,b_pairs,k_stage+1,b_regs[CUR^1],vmem_diag);
        consume_short<SLOT,false>(a_rb,b_regs[CUR][0],acc);
    }
    template <index_t CUR>
    CK_TILE_DEVICE void c_I2(index_t consumer_id,
                             index_t k_stage,
                             index_t num_loop,
                             const AReadBases& a_rb,
                             grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t b_rsrc,
                             const BPairs& b_pairs,
                             grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&b_regs)[2][2][Int8WaspM192RotatedAccTile::kNIters],
                             grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&a_regs)[Int8WaspM192RotatedAccTile::kMIters],
                             grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag& vmem_diag,
                             AccTile& acc) const
    {
        using namespace grouped_int8_nt_wasp_m192_rotated_detail;
        constexpr index_t SLOT = CUR * kAStageBytes;
        (void)consumer_id;

        if constexpr(!EarlyB)
            if(k_stage+1<num_loop)load_B_stage(b_rsrc,b_pairs,k_stage+1,b_regs[CUR^1],vmem_diag);
        consume_short<SLOT,true>(a_rb,b_regs[CUR][1],acc);
    }
    CK_TILE_DEVICE void consumer_run_c(index_t consumer_id,
                                                 index_t num_loop,
                                                 const AReadBases& a_rb,
                                                 grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t b_rsrc,
                                                 const BPairs& b_pairs,
                                                 grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&b_regs)[2][2][Int8WaspM192RotatedAccTile::kNIters],
                                                 grouped_int8_nt_wasp_m192_rotated_detail::stage16_t (&a_regs)[Int8WaspM192RotatedAccTile::kMIters],
                                                 grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag& vmem_diag,
                                                 AccTile& acc) const
    {
        for(index_t k = 0; k < num_loop; k += 2)
        {
            c_I1<0>(consumer_id, k, num_loop, b_rsrc, b_pairs, a_rb, b_regs, a_regs, vmem_diag, acc);
            if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
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
            grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            if(k + 1 < num_loop)
            {
                c_I1<1>(
                    consumer_id, k + 1, num_loop, b_rsrc, b_pairs, a_rb, b_regs, a_regs, vmem_diag, acc);
                if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
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
                grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            }
        }
    }
    CK_TILE_DEVICE static uint32_t record_stride(uint32_t physical_pitch)
    {
        if constexpr(RecordCap == 0)
            return physical_pitch;
        uint32_t stride = 1;
        static_for<0, 11, 1>{}([&](auto ii) {
            constexpr uint32_t width = 1u << decltype(ii)::value;
            if(physical_pitch >= width)
                stride = width;
        });
        return stride;
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

        grouped_int8_nt_wasp_m192_rotated_detail::WaspVmemDiag vmem_diag{};

        const uint32_t b_pitch_bytes = static_cast<uint32_t>(b_n_stride) * sizeof(ADataType);
        const uint32_t b_size_bytes =
            PackedB ? static_cast<uint32_t>(b_n_len) * b_pitch_bytes - static_cast<uint32_t>(b_n_origin) * 64u :
            static_cast<uint32_t>(b_n_len - b_n_origin) * b_pitch_bytes;
        const grouped_int8_nt_wasp_m192_rotated_detail::buffer_rsrc_t b_rsrc =
            grouped_int8_nt_wasp_m192_rotated_detail::wasp_make_buffer_rsrc(b_base, PackedB ? 0u : record_stride(b_pitch_bytes), b_size_bytes,
                                               !PackedB);

        grouped_int8_nt_wasp_m192_rotated_detail::wasp_sync_lds_maybe();

        if(wid < kNumProducerWaves)
        {

            const auto a_src = make_a_prod_src(a_base, lda, m_valid, wid);

            producer_fill(a_src, 0, wid);
            if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            grouped_int8_nt_wasp_m192_rotated_detail::wasp_wait_vmcnt0();
            grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();

            for(index_t k = 0; k + 1 < num_loop; ++k)
            {

                producer_fill(a_src, k + 1, wid);
#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
                if(k + 1 == grouped_int8_nt_wasp_m192_rotated_detail::kVmemSampleStage)
                {
                    vmem_diag.issued_at = grouped_int8_nt_wasp_m192_rotated_detail::wasp_memtime();
                    vmem_diag.issued = true;
                }
#endif
                if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
                grouped_int8_nt_wasp_m192_rotated_detail::wasp_wait_vmcnt0_producer(vmem_diag, k + 1);
                grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            }

            if(num_loop > 0)
            {
                if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
                grouped_int8_nt_wasp_m192_rotated_detail::wasp_wait_vmcnt0_producer(vmem_diag, num_loop);
                grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            }
        }
        else
        {

            acc = AccTile{};
            const index_t consumer_id = wid - kNumProducerWaves;

            grouped_int8_nt_wasp_m192_rotated_detail::stage16_t a_regs[Int8WaspM192RotatedAccTile::kMIters];

            const uint32_t lane    = static_cast<uint32_t>(get_lane_id());
            const uint32_t row     = lane & 15u;
            const uint32_t colg=lane>>4,wavebase=(row&3u)*1024u;
            const uint32_t common=(row>>2)*128u+colg*16u+(row&3u)*32u;
            const uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds))+wavebase;
            const AReadBases a_rb{base+common,base+((common+512u)&1023u),
                                  base+common+64u,base+((common+576u)&1023u)};
            const uint32_t nrow  = static_cast<uint32_t>(consumer_id) * 32u + row;
            const uint32_t inrow = colg * 16u;
            const uint32_t record_pitch = record_stride(b_pitch_bytes);
            const uint32_t quotient = b_pitch_bytes / record_pitch;
            const uint32_t remainder = b_pitch_bytes % record_pitch;
            // Low32 is the structured record index, high32 the byte offset.
            // index*record_pitch+offset equals row*physical_pitch+inrow.
            BPairs b_pr{
                (static_cast<uint64_t>(nrow * remainder + inrow) << 32) |
                    (nrow * quotient),
                (static_cast<uint64_t>((nrow + 16u) * remainder + inrow) << 32) |
                    ((nrow + 16u) * quotient)};

            if constexpr(PackedB)
            {
                b_pr.p0 = static_cast<uint32_t>(consumer_id) * 2048u + lane * 16u;
                b_pr.p1 = static_cast<uint32_t>(consumer_id) * 2048u + 1024u + lane * 16u;
                b_pr.half_stride = static_cast<uint32_t>(b_n_len) * 64u;
            }
            else if constexpr(BAddress != 0)
            {
                b_pr.p0 = nrow * b_pitch_bytes + inrow;
                b_pr.p1 = (nrow + 16u) * b_pitch_bytes + inrow;
            }
            grouped_int8_nt_wasp_m192_rotated_detail::stage16_t b_regs[2][2][Int8WaspM192RotatedAccTile::kNIters];

            if(num_loop > 0)
            {
                load_B_stage(b_rsrc, b_pr, 0, b_regs[0], vmem_diag);
            }
            if constexpr(!OneBarrier)grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();
            grouped_int8_nt_wasp_m192_rotated_detail::wasp_barrier_maybe();

            if constexpr(ConsumerSplit==8)consumer_run_c(consumer_id,num_loop,a_rb,b_rsrc,b_pr,b_regs,a_regs,vmem_diag,acc);
            else if(consumer_id<4)consumer_run_c(consumer_id,num_loop,a_rb,b_rsrc,b_pr,b_regs,a_regs,vmem_diag,acc);
            else consumer_run_cp(consumer_id,num_loop,a_rb,b_rsrc,b_pr,b_regs,a_regs,vmem_diag,acc);
        }

        grouped_int8_nt_wasp_m192_rotated_detail::wasp_sync_lds_maybe();

#if defined(CK_TILE_GROUPED_GEMM_I8_WASP_VMEM_WAIT_COUNT)
        if(get_lane_id() == 0)
        {
            const index_t cta = get_block_id();
            if(cta < 1024 && wid < 16)
            {
                grouped_int8_nt_wasp_m192_rotated_detail::g_wasp_vmem_samples[cta * 16 + wid] = vmem_diag.sample;
            }
        }
#endif

    }
};
}
