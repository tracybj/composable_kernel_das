// SPDX-License-Identifier: MIT
// Uses the documented raw INT8 MMAC/VMEM helpers of the current CK WASP pipeline.
#pragma once
#include "../../grouped_int8_nt_wasp_pipeline.hpp"

namespace ck_tile {
struct Int8WaspWideAccTile
{
    static constexpr index_t kMIters = 32;
    static constexpr index_t kNIters = 2;
    grouped_int8_nt_wasp_detail::frag_c_t v[kMIters * kNIters];
};

// Two512x64 A slots retain the existing64KB LDS budget. Four producer waves
// load128 rows each; eight consumer waves each own all512 rows and32 columns.
template<class Problem, index_t AReadBatch = 4, index_t RecordCap = 0>
struct GemmPipelineInt8WaspWide
    : GemmPipelineAgBgCrCompV3<Problem, GemmPipelineAgBgCrCompV4DefaultPolicy>
{
    using Shape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AccTile = Int8WaspWideAccTile;
    using Stage = grouped_int8_nt_wasp_detail::stage16_t;
    using Rsrc = grouped_int8_nt_wasp_detail::buffer_rsrc_t;
    static constexpr index_t BlockSize = 768;
    static constexpr index_t kStageK = 64;
    static constexpr index_t kAStageBytes = 32768;
    static constexpr index_t kWaspLdsBytes = 65536;
    static_assert(Shape::kM == 512 && Shape::kN == 256 && Shape::kK == 64);
    static_assert(Shape::NumWarps == 8);
    static_assert(AReadBatch == 2 || AReadBatch == 4 || AReadBatch == 8);
    static_assert(RecordCap == 0 || RecordCap == 1024);
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kWaspLdsBytes; }

    struct ASource { Rsrc rsrc; uint32_t offsets[8]; };
    struct BPairs { uint64_t p0; uint64_t p1; };
    using BStage = Stage[2];

    CK_TILE_DEVICE static uint32_t record_stride(uint32_t pitch)
    {
        if constexpr(RecordCap == 0) return pitch;
        uint32_t stride = 1;
        static_for<0, 11, 1>{}([&](auto ii) {
            constexpr uint32_t width = 1u << decltype(ii)::value;
            if(pitch >= width) stride = width;
        });
        return stride;
    }

    CK_TILE_DEVICE ASource make_a_source(const int8_t* a, index_t pitch, index_t wave) const
    {
        using namespace grouped_int8_nt_wasp_detail;
        const index_t lane = get_lane_id();
        const index_t row = lane >> 2;
        const index_t chunk = lane & 3;
        const uint32_t k_offset = static_cast<uint32_t>((chunk ^ (row & 3)) * 16);
        ASource result;
        result.rsrc = wasp_make_buffer_rsrc(a, pitch, 512u * pitch, false);
        static_for<0, 8, 1>{}([&](auto ii) {
            constexpr index_t i = decltype(ii)::value;
            result.offsets[i] = static_cast<uint32_t>(wave * 128 + i * 16 + row) * pitch + k_offset;
            asm volatile("" : "+v"(result.offsets[i]));
        });
        return result;
    }

    CK_TILE_DEVICE void produce(const ASource& src, index_t stage, index_t wave) const
    {
        using namespace grouped_int8_nt_wasp_detail;
        const uint32_t m0 = static_cast<uint32_t>((stage & 1) * kAStageBytes + wave * 8192);
        wasp_buffer_load_x8_to_lds(src.rsrc, m0, static_cast<uint32_t>(stage * kStageK),
                                 src.offsets[0], src.offsets[1], src.offsets[2], src.offsets[3],
                                 src.offsets[4], src.offsets[5], src.offsets[6], src.offsets[7]);
    }

    CK_TILE_DEVICE void load_b(Rsrc rsrc, const BPairs& p, index_t stage, BStage& b) const
    {
        const uint32_t offset = static_cast<uint32_t>(stage * kStageK);
        asm volatile("buffer_load_dwordx4 %0, %2, %4, %5 idxen offen\n\t"
                     "buffer_load_dwordx4 %1, %3, %4, %5 idxen offen\n"
                     : "=&v"(b[0].d4), "=&v"(b[1].d4)
                     : "v"(p.p0), "v"(p.p1), "s"(rsrc), "s"(offset)
                     : "memory");
    }

    template<index_t Offset>
    CK_TILE_DEVICE static void read_a(uint32_t base, Stage& value)
    {
        static_assert(Offset <= 65535);
        asm volatile("ds_read_b128 %0, %1 offset:%c2" : "=&v"(value.d4)
                     : "v"(base), "n"(Offset) : "memory");
    }

    template<index_t Slot>
    CK_TILE_DEVICE void consume(uint32_t a_base, const BStage& b, AccTile& acc) const
    {
        using namespace grouped_int8_nt_wasp_detail;
        Stage a[2][AReadBatch];
        static_for<0, AReadBatch, 1>{}([&](auto rr) {
            constexpr index_t i = decltype(rr)::value;
            read_a<Slot + i * 1024>(a_base, a[0][i]);
        });
        static_for<0, 32 / AReadBatch, 1>{}([&](auto gg) {
            constexpr index_t group = decltype(gg)::value;
            constexpr index_t slot = group & 1;
            constexpr index_t first = group * AReadBatch;
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            if constexpr(group + 1 < 32 / AReadBatch)
                static_for<0, AReadBatch, 1>{}([&](auto rr) {
                    constexpr index_t i = decltype(rr)::value;
                    read_a<Slot + (first + AReadBatch + i) * 1024>(a_base, a[slot ^ 1][i]);
                });
            static_for<0, 2, 1>{}([&](auto kk) {
                constexpr index_t k = decltype(kk)::value;
                static_for<0, AReadBatch, 1>{}([&](auto rr) {
                    constexpr index_t i = decltype(rr)::value;
                    static_for<0, 2, 1>{}([&](auto nn) {
                        constexpr index_t n = decltype(nn)::value;
                        acc.v[(first + i) * 2 + n] =
                            mmac_i8(a[slot][i].frag[k], b[n].frag[k], acc.v[(first + i) * 2 + n]);
                    });
                });
            });
        });
    }

    template<index_t Current>
    CK_TILE_DEVICE void consume_stage(index_t stage, index_t loops, uint32_t a_base,
                                      Rsrc rsrc, const BPairs& pairs,
                                      Stage (&b)[2][2], AccTile& acc) const
    {
        using namespace grouped_int8_nt_wasp_detail;
        wasp_wait_vmcnt0();
        if(stage + 1 < loops) load_b(rsrc, pairs, stage + 1, b[Current ^ 1]);
        consume<Current * kAStageBytes>(a_base, b[Current], acc);
        // Publish next A slot and release current A slot. All consumer waves
        // follow the same schedule; no LDS read remains live after this point.
        wasp_barrier_maybe();
    }

    CK_TILE_DEVICE void run_raw_into(const int8_t* a, index_t lda,
                                     const int8_t* b, index_t ldb, index_t b_rows,
                                     index_t loops, AccTile& acc, void* smem) const
    {
        using namespace grouped_int8_nt_wasp_detail;
        const index_t wave = get_warp_id();
        wasp_sync_lds_maybe();
        if(wave < 4)
        {
            const auto source = make_a_source(a, lda, wave);
            produce(source, 0, wave);
            wasp_wait_vmcnt0();
            wasp_barrier_maybe();
            for(index_t stage = 1; stage < loops; ++stage)
            {
                produce(source, stage, wave);
                wasp_wait_vmcnt0();
                wasp_barrier_maybe();
            }
            wasp_barrier_maybe();
        }
        else
        {
            acc = AccTile{};
            const uint32_t lane = get_lane_id();
            const uint32_t row = lane & 15u;
            const uint32_t col_group = lane >> 4;
            const uint32_t a_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem)) +
                                    row * 64u + (col_group ^ (row & 3u)) * 16u;
            const uint32_t nrow = static_cast<uint32_t>(wave - 4) * 32u + row;
            const uint32_t inrow = col_group * 16u;
            const uint32_t record = record_stride(ldb);
            const uint32_t q = ldb / record, rem = ldb % record;
            const BPairs pairs{
                (static_cast<uint64_t>(nrow * rem + inrow) << 32) | (nrow * q),
                (static_cast<uint64_t>((nrow + 16u) * rem + inrow) << 32) | ((nrow + 16u) * q)};
            const Rsrc rsrc = wasp_make_buffer_rsrc(b, record, static_cast<uint32_t>(b_rows) * ldb, true);
            Stage b_regs[2][2];
            load_b(rsrc, pairs, 0, b_regs[0]);
            wasp_barrier_maybe();
            for(index_t stage = 0; stage < loops; stage += 2)
            {
                consume_stage<0>(stage, loops, a_base, rsrc, pairs, b_regs, acc);
                if(stage + 1 < loops)
                    consume_stage<1>(stage + 1, loops, a_base, rsrc, pairs, b_regs, acc);
            }
        }
        wasp_sync_lds_maybe();
    }
};
} // namespace ck_tile
