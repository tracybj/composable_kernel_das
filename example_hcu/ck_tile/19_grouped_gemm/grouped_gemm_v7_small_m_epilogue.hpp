// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
//
// Independent small-M epilogue using the original V7 BF16 RNE conversion and
// 16-byte vector stores. Each wave stages its own 16-row slice in disjoint LDS.
// N must be a whole logical tile; only rows are predicated. For N=64 the row
// XOR mask is clamped to the four 16-byte units in a BF16 wave row.
#pragma once

#include "ck_tile/core.hpp"
#include "grouped_gemm_v7_uniform_epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m.hpp"

// The address arithmetic below is a port of the reference kernels, which are
// built with -Wno-old-style-cast; every cast is a fixed-width unsigned/signed
// conversion in an address expression. CK's own arch/buffer-addressing headers
// handle it the same way (see cast_pointer_to_constant_address_space). The
// 16 B store path reinterprets the output element pointer as one dwordx4
// vector -- an alignment-checked bit reinterpretation, not a type pun.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"

namespace ck_tile {



// ---------------------------------------------------------------------------
// The epilogue. `store_tile` is called with the accumulator the pipeline filled
// and the same LDS the ring used (which the k loop has released by then).
// ---------------------------------------------------------------------------
template <typename Element_O_, typename AccTile_, typename Geometry>
struct GroupedGemmV7SmallMStoreEpilogue
{
    using Element_O = remove_cvref_t<Element_O_>;
    using AccTile   = remove_cvref_t<AccTile_>;
    using G         = Geometry;

    // ---- framework-facing surface (what GroupedGemmKernel reads) --------
    using ODataType = Element_O;
    using DsDataType = ck_tile::tuple<>;   // no D tensors: single-stage store
    using DsLayout   = ck_tile::tuple<>;
    static constexpr index_t NumDTensor = 0;
    static constexpr memory_operation_enum MemoryOperation = memory_operation_enum::set;

    static constexpr int kE      = (int)sizeof(Element_O);       // 2 or 4 bytes
    static constexpr int kRowB   = (int)G::kWaveTileN * kE;      // wave row bytes
    static constexpr int kRowXorMask = (kRowB / 16 < 8 ? kRowB / 16 : 8) - 1;
    static constexpr int kUnits  = kRowB / 16;                   // 16-byte units per row
    static constexpr int kSliceB = 16 * kRowB;                   // one wave slab

    // The epilogue reuses the pipeline's ring buffer, so it asks for none of its
    // own: the framework's max(pipeline, epilogue) keeps the pipeline's 32 KB.
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    static_assert(kE == 2 || kE == 4, "2- or 4-byte output only");
    static_assert(kSliceB * (int)G::kWaves <= (int)G::kLdsBytes,
                  "the per-wave staging slabs must fit the ring LDS");

    // The store ALSO drains whatever loads the k tail left outstanding: their
    // LDS write-backs would otherwise land in the ring while the slabs are being
    // staged (observed as a few hundred bad elements, varying per run).
    CK_TILE_DEVICE static void store_tile(const V7UniformTileParams& p,
                                          const AccTile& acc,
                                          void* smem)
    {
        using namespace v7u_epi_detail;
        (void)smem;

        Element_O* d_out = reinterpret_cast<Element_O*>(p.d_ptr);
        const index_t d_stride = p.d_stride;

        const int tid  = (int)threadIdx.x;
        const int wid  = tid >> 6;
        const int lane = tid & 63;
        const int wave_m_start = (wid >> 1) * (int)G::kWaveTileM;
        const int wave_n_start = (wid & 1) * (int)G::kWaveTileN;

        v7u::wait_all_buffer_data_arrived<true>(0);

        const uint32_t epi_base = (uint32_t)(wid * kSliceB);
        const uint32_t row_q    = (uint32_t)(lane & 15);
        const uint32_t row_xor  = (uint32_t)(lane & kRowXorMask);
        const uint32_t col_q    = (uint32_t)(lane >> 4);

#pragma unroll
        for(index_t mF = 0; mF < G::kMTilesPerWave; ++mF)
        {
#pragma unroll
            for(index_t nF = 0; nF < G::kNTilesPerWave; ++nF)
            {
                const index_t idx = mF * G::kNTilesPerWave + nF;
#pragma unroll
                for(int c = 0; c < 4; ++c)
                {
                    const uint32_t byte = (col_q + (uint32_t)(nF * 16 + 4 * c)) * (uint32_t)kE;
                    const uint32_t addr = epi_base + row_q * (uint32_t)kRowB +
                                          (((byte >> 4) ^ row_xor) << 4) + (byte & 15u);
                    const Element_O val = fp32_to<Element_O>(acc.v[idx][c]);
                    if constexpr(kE == 2)
                    {
                        ds_write_ele(addr, *reinterpret_cast<const uint16_t*>(&val));
                    }
                    else
                    {
                        ds_write_ele(addr, *reinterpret_cast<const uint32_t*>(&val));
                    }
                }
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
            // Read the whole slice back FIRST, then store: the ds_read results
            // are inline-asm VGPR defs, so the compiler will not insert an LGKM
            // wait before the consuming global stores.
            constexpr int kJ = kSliceB / 64 / 16;   // vector reads per lane
            u32x4 vj[kJ];
#pragma unroll
            for(int j = 0; j < kJ; ++j)
            {
                const uint32_t t     = (uint32_t)lane + 64u * (uint32_t)j;
                const uint32_t r_in  = t / (uint32_t)kUnits;
                const uint32_t u     = t % (uint32_t)kUnits;
                const uint32_t addr  = epi_base + r_in * (uint32_t)kRowB +
                                       ((u ^ (r_in & (uint32_t)kRowXorMask)) << 4);
                vj[j] = ds_read_b128(addr);
            }
            asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
#pragma unroll
            for(int j = 0; j < kJ; ++j)
            {
                const uint32_t t    = (uint32_t)lane + 64u * (uint32_t)j;
                const uint32_t r_in = t / (uint32_t)kUnits;
                const uint32_t u    = t % (uint32_t)kUnits;
                const index_t row   = p.m_start + wave_m_start + mF * 16 + (index_t)r_in;
                if(row < p.m)
                {
                    const index_t col0 =
                        p.n_start + wave_n_start + (index_t)u * (16 / kE);
                    *reinterpret_cast<u32x4*>(&d_out[row * d_stride + col0]) = vj[j];
                }
            }
        }
    }
};

}   // namespace ck_tile

#pragma clang diagnostic pop
