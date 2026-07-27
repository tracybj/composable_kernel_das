// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/elementwise/element_wise_operation.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/utility/json_dump.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3_dsreadm.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3_w8_overlap.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3_gfx936_operand.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mls.hpp"

// #include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"


#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_PRESHUFFLE_V2 4
#define CK_TILE_PIPELINE_MLS 5
#define CK_TILE_PIPELINE_COMPUTE_V3_WARP_RAKED 6
#define CK_TILE_PIPELINE_COMPUTE_V3_DSREADM 7
#define CK_TILE_PIPELINE_COMPUTE_V3_W8_OVERLAP 8
#define CK_TILE_PIPELINE_COMPUTE_V3_TRITON_OPERAND 9
#define CK_TILE_PIPELINE_COMPUTE_V3_DSREADM_STAGE 10

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
    if constexpr(M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile_flatmm()
{
#if defined(CK_GFX950_SUPPORT)
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 64;
    else
        return sizeof(PrecType) == 2 ? 32 : 128;
#else
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 32;
    else
        return sizeof(PrecType) == 2 ? 32 : 64;
#endif
}

template <typename DataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using CDataType   = ck_tile::bf16_t;
    using AccDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::int8_t>
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
    // using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::pk_int4_t>
{
    using ADataType   = ck_tile::pk_int4_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
};

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    // static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    // static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
    static constexpr ck_tile::index_t TileParitionerM01      = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t Pipeline      = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
    static constexpr bool Persistent                = true;
    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr bool SupportsFastRowMajorB     = false;
    static constexpr bool SupportsFastColumnMajorA  = false;
    static constexpr bool SupportsSplitKOutputAtomic = false;
    static constexpr bool TiledMMAPermuteN           = false;
    // Zero keeps CShuffle's legacy default: repeat = warp tile / 16, interleave = 1.
    static constexpr ck_tile::index_t CShuffleWarpGemmMRepeat     = 0;
    static constexpr ck_tile::index_t CShuffleWarpGemmNRepeat     = 0;
    static constexpr ck_tile::index_t CShuffleWarpGemmMInterleave = 0;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 0;
};

template <typename PrecType>
struct GemmConfigComputeV3_2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    // static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
    static constexpr bool SupportsFastRowMajorB = true;
    static constexpr bool SupportsFastColumnMajorA = true;
};

// Legacy-named V4 padded variant. Keep the type name for source compatibility, but pad all
// logical GEMM dimensions so dynamic grouped problems can use arbitrary M/N/K tile tails.
template <typename PrecType>
struct GemmConfigComputeV4Mpad : public GemmConfigComputeV4<PrecType>
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

// Tuning candidate for large, regular FP16/BF16 GEMMs. Keep it separate from the production
// V4 config until same-session A/B measurements and all layout/padding regressions pass.
//
// The 128x128 block tile increases A/B reuse. FP16 uses the existing 32x64 MMAC warp dispatcher;
// BF16 can use either the original 16x64 dispatcher or the 32x64 dispatcher added for the
// square-wide candidate.
template <typename PrecType>
struct GemmConfigComputeV4Square : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile =
        std::is_same_v<PrecType, ck_tile::half_t> ? 32 : 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;
};

template <typename PrecType>
struct GemmConfigComputeV4SquareMpad : public GemmConfigComputeV4Square<PrecType>
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

// gfx936/gfx938 spill-reduction config for the 128x128 FP16/BF16 square path.
// Split M across four waves instead of doing two M iterations in each of two waves.
// This keeps the block tile, K tile, LDS buffering and per-wave MMAC primitive unchanged,
// while halving each wave's C accumulator footprint.
template <typename PrecType>
struct GemmConfigComputeV4SquareM4 : public GemmConfigComputeV4Square<PrecType>
{
    static constexpr ck_tile::index_t M_Warp      = 4;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
};

template <typename PrecType>
struct GemmConfigComputeV4SquareM4Mpad : public GemmConfigComputeV4SquareM4<PrecType>
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

// Dynamic grouped GEMM often has device-resident per-group M while host-visible
// N/K are tile aligned. Preserve M-tail safety without paying the N/K padding
// cost in that bounded contract.
template <typename PrecType>
struct GemmConfigComputeV4SquareM4MOnlyPad
    : public GemmConfigComputeV4SquareM4<PrecType>
{
    static constexpr bool kPadM = true;
};

// gfx936-selected locality mapping matching the persistent M-group swizzle used
// by the Triton reference while leaving tile, pipeline and resources fixed.
template <typename PrecType>
struct GemmConfigComputeV4SquareM4M01_4
    : public GemmConfigComputeV4SquareM4<PrecType>
{
    static constexpr ck_tile::index_t TileParitionerM01 = 4;
};

// gfx936 TN backward prefers hardware-scheduled full-grid traversal. Keep
// M01=4 because it remained the best TN mapping in the persistent A/B matrix.
template <typename PrecType>
struct GemmConfigComputeV4SquareM4M01_4NonPersistent
    : public GemmConfigComputeV4SquareM4M01_4<PrecType>
{
    static constexpr bool Persistent = false;
};

// gfx938-only direct MLS candidate. Keep it isolated from W1 until correctness,
// ISA and same-interface A/B validation are complete.
template <typename PrecType>
struct GemmConfigMls : public GemmConfigBase
{
    static_assert(std::is_same_v<PrecType, ck_tile::half_t> ||
                      std::is_same_v<PrecType, ck_tile::bf16_t>,
                  "The direct grouped-GEMM MLS config supports FP16 and BF16");

    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
    static constexpr bool Persistent = true;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool TiledMMAPermuteN = true;
    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;
    static constexpr bool SupportsSplitKOutputAtomic = true;
    static constexpr ck_tile::index_t CShuffleWarpGemmMRepeat     = 1;
    static constexpr ck_tile::index_t CShuffleWarpGemmNRepeat     = 1;
    static constexpr ck_tile::index_t CShuffleWarpGemmMInterleave = 2;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 2;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_MLS;
    static constexpr int kBlockPerCu = 1;
};

// Register-prefetch candidate for large gfx938 FP16/BF16 grouped GEMMs.
//
// Compute V3 keeps one 256x256x64 A/B stage in LDS (exactly 64 KiB for
// 16-bit inputs), while prefetching the following K64 stage into registers.
// This is the smallest existing CK pipeline that matches Triton's decisive
// K64/two-stage behavior without requiring an unsupported 128 KiB LDS
// allocation. Keep it behind an explicit tuning target until correctness,
// resource usage, and same-session performance have all been validated.
template <typename PrecType>
struct GemmConfigComputeV3RegPrefetch256 : public GemmConfigBase
{
    static_assert(std::is_same_v<PrecType, ck_tile::half_t> ||
                      std::is_same_v<PrecType, ck_tile::bf16_t>,
                  "The grouped-GEMM register-prefetch candidate supports FP16/BF16");

    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
    static constexpr bool Persistent       = true;
    static constexpr bool DoubleSmemBuffer = false;
    // Bring up the exact Primus forward layout first. Enabling NN/TN here
    // instantiates additional LDS descriptor paths and is intentionally
    // deferred until the RC candidate has passed its performance gate.
    static constexpr bool SupportsFastRowMajorB    = false;
    static constexpr bool SupportsFastColumnMajorA = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV3RegPrefetch256W16Group8Aligned
    : public GemmConfigComputeV3RegPrefetch256<PrecType>
{
    // gfx936 selected path: the Triton-sized 256x256x64 tile is distributed
    // over sixteen waves to avoid the eight-wave V3 register spill.
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr bool kPadM               = false;
    static constexpr bool kPadN               = false;
    static constexpr bool kPadK               = false;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr ck_tile::index_t Pipeline =
        CK_TILE_PIPELINE_COMPUTE_V3_WARP_RAKED;
};

// Independent gfx936 W8 resource experiment. The block shape matches the
// Triton-sized 256x256x64 tile, but the pipeline rolls through two K16 slots
// so accumulator, LDS operands and global prefetches do not all retain K64
// lifetimes. Keep this explicit-target-only until correctness, ISA and
// zero-spill gates pass and it beats the selected W16 target.
template <typename PrecType>
struct GemmConfigComputeV3W8Overlap256Aligned
    : public GemmConfigComputeV3RegPrefetch256<PrecType>
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr int kBlockPerCu = 2;
    static constexpr ck_tile::index_t Pipeline =
        CK_TILE_PIPELINE_COMPUTE_V3_W8_OVERLAP;
};

// Phase-39 isolated operand-mapping candidate. Keep Triton's 256x256x64,
// 2x4-wave geometry, but expose the native 16x16x16 MMAC tile to the block
// layer. The resulting MRepeat=8/NRepeat=4 outer loops interleave the wave
// coordinate between repeats exactly like Triton's linear layout instead of
// assigning each wave one contiguous M128xN64 rectangle.
template <typename PrecType>
struct GemmConfigComputeV3TritonOperand256Aligned
    : public GemmConfigComputeV3RegPrefetch256<PrecType>
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;
    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr int kBlockPerCu = 1;
    static constexpr ck_tile::index_t Pipeline =
        CK_TILE_PIPELINE_COMPUTE_V3_TRITON_OPERAND;
};

// Exact Primus Triton program-id traversal for the 4096x7168 tile grid.
// GroupNum=1 removes CK's outer eight-way remap, while M01=4 preserves the
// four-M-tile locality used by Triton's GROUP_SIZE_M=4 mapping.
template <typename PrecType>
struct GemmConfigComputeV3TritonOperand256ExactTraversal
    : public GemmConfigComputeV3TritonOperand256Aligned<PrecType>
{
#if defined(CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM)
    static constexpr ck_tile::index_t TileParitionerGroupNum =
        CK_TILE_GROUPED_GEMM_TRITON_TILE_GROUP_NUM;
#else
    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
#endif
#if defined(CK_TILE_GROUPED_GEMM_TRITON_TILE_M01)
    static constexpr ck_tile::index_t TileParitionerM01 =
        CK_TILE_GROUPED_GEMM_TRITON_TILE_M01;
#else
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
#endif
};

// gfx936 NN/TN large-shape path. It keeps the selected V3 geometry but uses
// the original thread-raked distribution: applying warp-raked A/B to the
// transpose paths violates their vector offset/alignment contract.
template <typename PrecType>
struct GemmConfigComputeV3RegPrefetch256W16Group8DefaultPolicyAllLayouts
    : public GemmConfigComputeV3RegPrefetch256W16Group8Aligned<PrecType>
{
    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;
    static constexpr ck_tile::index_t Pipeline     = CK_TILE_PIPELINE_COMPUTE_V3;
};

// Explicit gfx936 FP16/BF16 NN/TN probe. It keeps the validated V3 scheduler and
// 256x256x64 geometry, but consumes packed LDS tiles with ds_read_m32x16.
template <typename PrecType>
struct GemmConfigComputeV3RegPrefetch256W16Group8Dsreadm
    : public GemmConfigComputeV3RegPrefetch256W16Group8DefaultPolicyAllLayouts<PrecType>
{
    static_assert(ck_tile::is_any_of<PrecType, ck_tile::half_t, ck_tile::bf16_t>::value,
                  "CK Tile dsreadm grouped GEMM supports FP16/BF16");
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3_DSREADM;
};

// Selected gfx936 large aligned NN/TN configuration. Eight waves own a
// 256x256x64 block while the DSReadM pipeline retains only two K16 operand
// stages at a time. The older W16 DSReadM config remains available as an
// independently reproducible fallback.
template <typename PrecType>
struct GemmConfigComputeV3DsreadmStage256W8
    : public GemmConfigComputeV3RegPrefetch256<PrecType>
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;
    static constexpr bool SupportsFastRowMajorB    = true;
    static constexpr bool SupportsFastColumnMajorA = true;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr int kBlockPerCu = 1;
    static constexpr ck_tile::index_t Pipeline =
        CK_TILE_PIPELINE_COMPUTE_V3_DSREADM_STAGE;
};

template <typename PrecType>
struct GemmConfigMlsK64 : public GemmConfigMls<PrecType>
{
    // Keep the proven 128x128/block256 geometry, but halve the number of
    // pipeline iterations. Two 128x64 A/B stages consume exactly 64 KiB LDS.
    static constexpr ck_tile::index_t K_Tile = 64;
};

// Large-tile MLS config for the 4096x7168x4096, group=4 persistent workload.
// The original two-stage pipeline uses 64 KiB LDS; the validated single-stage
// default uses 32 KiB and permits two resident blocks/CU on gfx938.
template <typename PrecType>
struct GemmConfigMls256 : public GemmConfigMls<PrecType>
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;

    static constexpr ck_tile::index_t CShuffleWarpGemmMRepeat     = 1;
    // Keep the validated per-wave MR1/NR1/MI2/NI2 MMAC output mapping.
    // CShuffle's space-filling curve iterates over the larger block tile.
    static constexpr ck_tile::index_t CShuffleWarpGemmNRepeat     = 1;
    static constexpr ck_tile::index_t CShuffleWarpGemmMInterleave = 2;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 2;
#if defined(CK_TILE_GROUPED_GEMM_MLS_SINGLE_STAGE)
    // The K32 single-stage pipeline consumes 32 KiB LDS and was validated with
    // two resident blocks/CU on gfx938. The legacy two-stage path still needs
    // 64 KiB and keeps the conservative one-block contract.
    static constexpr int kBlockPerCu = 2;
#else
    static constexpr int kBlockPerCu = 1;
#endif
};

template <typename PrecType>
struct GemmConfigMls256CShuffleMR2 : public GemmConfigMls256<PrecType>
{
    static constexpr ck_tile::index_t CShuffleWarpGemmMRepeat     = 2;
    static constexpr ck_tile::index_t CShuffleWarpGemmMInterleave = 1;
};

template <typename PrecType>
struct GemmConfigMls256CShuffleNR2 : public GemmConfigMls256<PrecType>
{
    static constexpr ck_tile::index_t CShuffleWarpGemmNRepeat     = 2;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 1;
};

template <typename PrecType>
struct GemmConfigMls256CShuffleMR2NR2 : public GemmConfigMls256CShuffleMR2<PrecType>
{
    static constexpr ck_tile::index_t CShuffleWarpGemmNRepeat     = 2;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 1;
};

template <typename PrecType>
struct GemmConfigMls256NonPersistent : public GemmConfigMls256<PrecType>
{
    // Ablate the persistent tile loop without changing the MLS compute body.
    static constexpr bool Persistent = false;
};

template <typename PrecType>
struct GemmConfigMls256K64 : public GemmConfigMls256<PrecType>
{
    static constexpr ck_tile::index_t K_Tile = 64;
};

template <typename PrecType>
struct GemmConfigMls256x128 : public GemmConfigMls256<PrecType>
{
    static constexpr ck_tile::index_t N_Tile = 128;
};

template <typename PrecType>
struct GemmConfigMls128x256 : public GemmConfigMls256<PrecType>
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
};

template <typename PrecType>
struct GemmConfigMls256N4 : public GemmConfigMls256<PrecType>
{
    // Same tile, LDS, block size and total wave count as the W8 baseline.
    // Reorient the wave grid toward N for the 4096x7168 target.
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 4;
};

template <typename PrecType>
struct GemmConfigMls256Spatial : public GemmConfigMls256<PrecType>
{
    // Remap the persistent tile stream into 8 coarse groups with 4 adjacent
    // M tiles per local column. This keeps geometry/codegen constant and
    // changes only cross-CTA spatial locality.
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
};

template <typename PrecType>
struct GemmConfigMls256M01_4 : public GemmConfigMls256<PrecType>
{
    // Match Triton's GROUP_SIZE_M=4 traversal exactly: no coarse GroupNum
    // remap, and four consecutive M tiles share one N/B tile.
    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
};

template <typename PrecType>
struct GemmConfigMls256W16 : public GemmConfigMls256<PrecType>
{
    // Sixteen waves halve the per-wave C accumulator footprint while keeping
    // the same 256x256x32 block tile and 64 KiB MLS staging allocation.
    static constexpr ck_tile::index_t N_Warp = 4;
};

template <typename PrecType>
struct GemmConfigMls256x512W16 : public GemmConfigMls256<PrecType>
{
    // A single 256x512x32 LDS stage occupies 48 KiB. Sixteen waves retain
    // the same eight 32x32 accumulator tiles per wave as MLS256/W8 while
    // doubling the persistent CTA's output area.
    static constexpr ck_tile::index_t N_Tile = 512;
    static constexpr ck_tile::index_t N_Warp = 4;
};

template <typename PrecType>
struct GemmConfigMls256Warp32x64 : public GemmConfigMls256<PrecType>
{
    // Preserve the same 256x256 block, eight waves and per-wave output area,
    // but consume two adjacent 16x16 MMAC columns in one 32x64 warp tile.
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t CShuffleWarpGemmNInterleave = 4;
};

template <typename AccDataType_, typename ODataType_>
struct GroupedGemmDirectStoreEpilogue
{
    using AccDataType = ck_tile::remove_cvref_t<AccDataType_>;
    using ODataType   = ck_tile::remove_cvref_t<ODataType_>;
    using DsDataType  = ck_tile::tuple<>;
    using DsLayout    = ck_tile::tuple<>;

    static constexpr ck_tile::index_t NumDTensor = 0;
    static constexpr ck_tile::memory_operation_enum MemoryOperation =
        ck_tile::memory_operation_enum::set;

    // Triton's gfx936 epilogue converts one 128x256 half-output slab at a time
    // through the 64-KiB compute LDS. Doing the conversion after the K loop
    // lets this candidate reuse the A/B allocation instead of requesting the
    // 128 KiB needed by a full 256x256 half CShuffle tile.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return 128 * 256 * sizeof(ODataType);
    }

    // The blockwise LDS read gives every lane eight consecutive half values,
    // matching Triton's buffer_store_dwordx4 output granularity.
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetVectorSizeC() { return 8; }

    template <ck_tile::index_t I>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t
    GetVectorSizeD(ck_tile::number<I>)
    {
        return 1;
    }

    using Vec4 = ck_tile::ext_vector_t<ODataType, 4>;
    using Vec8 = ck_tile::ext_vector_t<ODataType, 8>;

    template <ck_tile::index_t I0,
              ck_tile::index_t I1,
              ck_tile::index_t I2,
              ck_tile::index_t I3,
              typename CBuffer>
    CK_TILE_DEVICE static Vec4 MakeAccVec4(const CBuffer& c)
    {
        return Vec4{ck_tile::type_convert<ODataType>(c[ck_tile::number<I0>{}]),
                    ck_tile::type_convert<ODataType>(c[ck_tile::number<I1>{}]),
                    ck_tile::type_convert<ODataType>(c[ck_tile::number<I2>{}]),
                    ck_tile::type_convert<ODataType>(c[ck_tile::number<I3>{}])};
    }

    CK_TILE_DEVICE static Vec8 MakeOutputVec8(const Vec4& x0,
                                              const Vec4& x1,
                                              const Vec4& x2,
                                              const Vec4& x3,
                                              ck_tile::number<0>)
    {
        return Vec8{x0[0], x1[0], x2[0], x3[0], x0[1], x1[1], x2[1], x3[1]};
    }

    CK_TILE_DEVICE static Vec8 MakeOutputVec8(const Vec4& x0,
                                              const Vec4& x1,
                                              const Vec4& x2,
                                              const Vec4& x3,
                                              ck_tile::number<1>)
    {
        return Vec8{x0[2], x1[2], x2[2], x3[2], x0[3], x1[3], x2[3], x3[3]};
    }

    template <ck_tile::index_t Offset0, ck_tile::index_t Offset1>
    CK_TILE_DEVICE static void StoreLdsPairSt64(ck_tile::index_t byte_address,
                                                const Vec4& x0,
                                                const Vec4& x1)
    {
        static_assert(sizeof(Vec4) == 8,
                      "ds_write2st64_b64 requires two 64-bit payloads");
        asm volatile("ds_write2st64_b64 %0, %1, %2 offset0:%3 offset1:%4"
                     :
                     : "v"(byte_address),
                       "v"(ck_tile::bit_cast<ck_tile::fp32x2_t>(x0)),
                       "v"(ck_tile::bit_cast<ck_tile::fp32x2_t>(x1)),
                       "n"(Offset0),
                       "n"(Offset1)
                     : "memory");
    }

    template <ck_tile::index_t RawBase, typename CBuffer>
    CK_TILE_DEVICE static void StoreAccRoundToLds(const CBuffer& c_buffer)
    {
        const Vec4 s0 = MakeAccVec4<RawBase + 0,
                                    RawBase + 1,
                                    RawBase + 16,
                                    RawBase + 17>(c_buffer);
        const Vec4 s1 = MakeAccVec4<RawBase + 2,
                                    RawBase + 3,
                                    RawBase + 18,
                                    RawBase + 19>(c_buffer);
        const Vec4 s2 = MakeAccVec4<RawBase + 8,
                                    RawBase + 9,
                                    RawBase + 24,
                                    RawBase + 25>(c_buffer);
        const Vec4 s3 = MakeAccVec4<RawBase + 10,
                                    RawBase + 11,
                                    RawBase + 26,
                                    RawBase + 27>(c_buffer);
        const Vec4 s4 = MakeAccVec4<RawBase + 4,
                                    RawBase + 5,
                                    RawBase + 20,
                                    RawBase + 21>(c_buffer);
        const Vec4 s5 = MakeAccVec4<RawBase + 6,
                                    RawBase + 7,
                                    RawBase + 22,
                                    RawBase + 23>(c_buffer);
        const Vec4 s6 = MakeAccVec4<RawBase + 12,
                                    RawBase + 13,
                                    RawBase + 28,
                                    RawBase + 29>(c_buffer);
        const Vec4 s7 = MakeAccVec4<RawBase + 14,
                                    RawBase + 15,
                                    RawBase + 30,
                                    RawBase + 31>(c_buffer);

        const ck_tile::index_t tid = ck_tile::get_thread_id();
        const ck_tile::index_t store_byte_offset =
            (((tid & 14) << 9) | ((tid & 48) << 3) |
             (((tid & 14) << 2) ^ ((tid & 192) >> 3)) |
             ((tid & 1) << 13) | ((tid >> 2) & 64));
        const ck_tile::index_t store_byte_offset_xor32 =
            store_byte_offset ^ 32;

        StoreLdsPairSt64<0, 32>(store_byte_offset, s0, s1);
        StoreLdsPairSt64<1, 33>(store_byte_offset, s2, s3);
        StoreLdsPairSt64<0, 32>(store_byte_offset_xor32, s4, s5);
        StoreLdsPairSt64<1, 33>(store_byte_offset_xor32, s6, s7);
    }

    template <typename AccTile>
    CK_TILE_DEVICE static void PreStoreRound0(const AccTile& acc_tile, void*)
    {
        const auto& c_buffer = acc_tile.get_thread_buffer();
        static_assert(ck_tile::remove_cvref_t<AccTile>::get_thread_buffer_size() == 128,
                      "fused round-0 store requires 128 accumulators per lane");
        StoreAccRoundToLds<0>(c_buffer);
    }

    CK_TILE_DEVICE static void SyncAfterEpilogueLdsWrite()
    {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_LDS_ONLY_SYNC)
        asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                     "s_barrier"
                     :
                     :
                     : "memory");
#else
        ck_tile::block_sync_lds();
#endif
    }

    CK_TILE_DEVICE static void SyncBeforeEpilogueLdsOverwrite()
    {
        SyncAfterEpilogueLdsWrite();
    }

    template <typename OWindow, typename AccTile, typename DWindow>
    CK_TILE_DEVICE void operator()(OWindow& o_window,
                                   const AccTile& acc_tile,
                                   const DWindow&,
                                   void* p_smem = nullptr) const
    {
        const auto& c_buffer = acc_tile.get_thread_buffer();
        static_assert(ck_tile::remove_cvref_t<AccTile>::get_thread_buffer_size() == 128,
                      "Triton-layout epilogue requires 128 accumulators per lane");

        // This is the actual Triton gfx936 accumulator-to-output mapping for a
        // 256x256 block with warpsPerCTA=[2,4]. Each round converts only 32 of
        // the 128 FP32 accumulators, writes eight half4 vectors into LDS, then
        // reads the transposed layout and emits four half8 global stores.
        constexpr auto lds_desc = ck_tile::make_naive_tensor_descriptor_packed(
            ck_tile::make_tuple(ck_tile::number<32768>{}), ck_tile::number<8>{});
        auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_desc);

        const ck_tile::index_t tid = ck_tile::get_thread_id();
        const ck_tile::index_t load_byte_offset =
            (((tid & 1) << 14) | ((tid << 5) & 512) | ((tid << 8) & 8192) |
             ((((tid & 448) << 4) | ((tid & 14) << 2)) ^ ((tid & 448) >> 3)));

        const auto block_origin = o_window.get_window_origin();
        const ck_tile::index_t output_col =
            block_origin.at(ck_tile::number<1>{}) + (tid & 31) * 8;
        const ck_tile::index_t output_row_lane =
            block_origin.at(ck_tile::number<0>{}) + ((tid >> 5) & 15);
        auto o_view = o_window.get_bottom_tensor_view();
        constexpr bool kStoreOobCheck = true;

        constexpr auto raw_bases = ck_tile::sequence<0, 64, 32, 96>{};
        constexpr auto row_bases = ck_tile::sequence<0, 128, 64, 192>{};

        ck_tile::static_for<0, 4, 1>{}([&](auto i_round) {
            constexpr ck_tile::index_t raw_base = raw_bases.at(i_round);
            constexpr ck_tile::index_t row_base = row_bases.at(i_round);

#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
            if constexpr(i_round == ck_tile::number<0>{})
            {
                // Round 0 was converted and written after the first final-K16
                // M pair. Its LDS traffic overlapped the remaining 24 MMACs.
                SyncAfterEpilogueLdsWrite();
            }
            else
            {
                SyncBeforeEpilogueLdsOverwrite();
                StoreAccRoundToLds<raw_base>(c_buffer);
                SyncAfterEpilogueLdsWrite();
            }
#else
#if defined(CK_TILE_GROUPED_GEMM_TRITON_EPILOGUE_PRE_SYNC)
            if constexpr(i_round != ck_tile::number<0>{})
            {
                SyncBeforeEpilogueLdsOverwrite();
            }
#else
            SyncBeforeEpilogueLdsOverwrite();
#endif

            // For b64 st64 operations one immediate step is 512 bytes. The
            // exact Triton layout places each producer pair 16 KiB apart, so
            // the eight half4 stores become four explicit write2 operations
            // with offsets [0,32] and [1,33].
            StoreAccRoundToLds<raw_base>(c_buffer);
            SyncAfterEpilogueLdsWrite();
#endif

            constexpr ck_tile::index_t load_offsets[] =
                {0, 128, 256, 384, 64, 192, 320, 448};
#if defined(CK_TILE_GROUPED_GEMM_GFX936_EPILOGUE_SPLIT_READ_STORE)
            const auto store_output = [&](const Vec8& output, auto i_row) {
                const auto coord = ck_tile::make_tensor_coordinate(
                    o_view.get_tensor_descriptor(),
                    ck_tile::array<ck_tile::index_t, 2>{
                        output_row_lane + row_base + i_row * 16,
                        output_col});
                const auto& output_buffer =
                    reinterpret_cast<
                        const ck_tile::thread_buffer<ODataType, 8>&>(
                        output);
                o_view
                    .template set_vectorized_elements_raw<
                        ck_tile::thread_buffer<ODataType, 8>,
                        kStoreOobCheck>(coord, 0, output_buffer);
            };

            Vec4 first_loads[4];
            ck_tile::static_for<0, 4, 1>{}([&](auto i) {
                const auto coord = ck_tile::make_tensor_coordinate(
                    lds_desc,
                    ck_tile::array<ck_tile::index_t, 1>{
                        (load_byte_offset + load_offsets[i]) /
                        static_cast<ck_tile::index_t>(
                            sizeof(ODataType))});
                first_loads[i] =
                    lds_view.template get_vectorized_elements<Vec4, false>(
                        coord, 0);
            });
            const Vec8 output0 =
                MakeOutputVec8(first_loads[0],
                               first_loads[1],
                               first_loads[2],
                               first_loads[3],
                               ck_tile::number<0>{});
            const Vec8 output2 =
                MakeOutputVec8(first_loads[0],
                               first_loads[1],
                               first_loads[2],
                               first_loads[3],
                               ck_tile::number<1>{});
            store_output(output0, ck_tile::number<0>{});
            store_output(output2, ck_tile::number<2>{});

            Vec4 second_loads[4];
            ck_tile::static_for<0, 4, 1>{}([&](auto i) {
                const auto coord = ck_tile::make_tensor_coordinate(
                    lds_desc,
                    ck_tile::array<ck_tile::index_t, 1>{
                        (load_byte_offset + load_offsets[i + 4]) /
                        static_cast<ck_tile::index_t>(
                            sizeof(ODataType))});
                second_loads[i] =
                    lds_view.template get_vectorized_elements<Vec4, false>(
                        coord, 0);
            });
            const Vec8 output1 =
                MakeOutputVec8(second_loads[0],
                               second_loads[1],
                               second_loads[2],
                               second_loads[3],
                               ck_tile::number<0>{});
            const Vec8 output3 =
                MakeOutputVec8(second_loads[0],
                               second_loads[1],
                               second_loads[2],
                               second_loads[3],
                               ck_tile::number<1>{});
            store_output(output1, ck_tile::number<1>{});
            store_output(output3, ck_tile::number<3>{});
#else
            Vec4 loads[8];
            ck_tile::static_for<0, 8, 1>{}([&](auto i) {
                const auto coord = ck_tile::make_tensor_coordinate(
                    lds_desc,
                    ck_tile::array<ck_tile::index_t, 1>{
                        (load_byte_offset + load_offsets[i]) /
                        static_cast<ck_tile::index_t>(sizeof(ODataType))});
                loads[i] =
                    lds_view.template get_vectorized_elements<Vec4, false>(coord, 0);
            });

            const Vec8 outputs[4] = {
                MakeOutputVec8(loads[0], loads[1], loads[2], loads[3], ck_tile::number<0>{}),
                MakeOutputVec8(loads[4], loads[5], loads[6], loads[7], ck_tile::number<0>{}),
                MakeOutputVec8(loads[0], loads[1], loads[2], loads[3], ck_tile::number<1>{}),
                MakeOutputVec8(loads[4], loads[5], loads[6], loads[7], ck_tile::number<1>{})};

            ck_tile::static_for<0, 4, 1>{}([&](auto i) {
                const auto coord = ck_tile::make_tensor_coordinate(
                    o_view.get_tensor_descriptor(),
                    ck_tile::array<ck_tile::index_t, 2>{
                        output_row_lane + row_base + i * 16, output_col});
                const auto& output_buffer =
                    reinterpret_cast<
                        const ck_tile::thread_buffer<ODataType, 8>&>(outputs[i]);
                o_view
                    .template set_vectorized_elements_raw<
                        ck_tile::thread_buffer<ODataType, 8>,
                        kStoreOobCheck>(coord, 0, output_buffer);
            });
#endif
        });

        ck_tile::buffer_store_fence();
    }
};

template <typename PrecType>
struct GemmConfigComputeV4SquareWide : public GemmConfigComputeV4Square<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
};

template <typename PrecType>
struct GemmConfigComputeV4SquareWideMpad : public GemmConfigComputeV4SquareWide<PrecType>
{
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;
};

// V5 variant with kPadM=true for fp8 MOE dynamic-M (fixed N/K).
template <typename PrecType>
struct GemmConfigComputeV5Mpad : public GemmConfigComputeV5<PrecType>
{
    static constexpr bool kPadM = true;
};

template <typename PrecType>
struct GemmConfigComputeFp8N64K64 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8V2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8K64 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

// Bf8K64 variant with kPadM=true for bf8 MOE dynamic-M (fixed N/K).
template <typename PrecType>
struct GemmConfigComputeBf8K64Mpad : public GemmConfigComputeBf8K64<PrecType>
{
    static constexpr bool kPadM = true;
};

template <typename PrecType>
struct GemmConfigComputeV6 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV7 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile_flatmm<PrecType, M_Warp_Tile>();

    static constexpr bool kPadK = true;

    static constexpr int kBlockPerCu           = 1;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile_flatmm<PrecType, M_Warp_Tile>();

    static constexpr int kBlockPerCu           = 2;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr bool kPadK                = true;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode_Wmma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 32 / sizeof(PrecType);
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadK = true;

    static constexpr int kBlockPerCu           = 1;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
};

template <ck_tile::index_t PipelineId>
struct PipelineTypeTraits;

// template <>
// struct PipelineTypeTraits<CK_TILE_PIPELINE_MEMORY>
// {
//     template <typename PipelineProblem>
//     using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
//     template <typename PipelineProblem>
//     using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
// };

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3_WARP_RAKED>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3<
            PipelineProblem,
            ck_tile::GemmPipelineAgBgCrCompV3WarpRakedPolicy>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3_DSREADM>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3Dsreadm<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3_DSREADM_STAGE>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3Dsreadm<PipelineProblem, true>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline =
        ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3_W8_OVERLAP>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3W8Overlap<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline =
        ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3_TRITON_OPERAND>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompV3TritonOperand<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline =
        ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_MLS>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMls<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::GemmPipelineAgBgCrMls<PipelineProblem>;
};

template <typename PrecType>
struct GemmConfigComputeInt4 : public GemmConfigBase
{
    // Production config (WT32x64x64 MR2NR4, same shape as GemmConfigComputeV5)
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);  // 128 pk_int4 = 256 logical

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

// template <>
// struct PipelineTypeTraits<CK_TILE_PIPELINE_PRESHUFFLE_V2>
// {
//     template <typename PipelineProblem>
//     using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
//     template <typename PipelineProblem>
//     using UniversalGemmPipeline =
//         ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
// };

using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs;

#ifndef CK_GROUPED_GEMM_ABI_DEFINED
#define CK_GROUPED_GEMM_ABI_DEFINED

extern "C" {

enum ck_tile_hcu_grouped_gemm_dtype
{
    CK_TILE_HCU_GROUPED_GEMM_FP16 = 0,
    CK_TILE_HCU_GROUPED_GEMM_FP8  = 1,
    CK_TILE_HCU_GROUPED_GEMM_INT8 = 2,
    CK_TILE_HCU_GROUPED_GEMM_BF8  = 3,
    CK_TILE_HCU_GROUPED_GEMM_BF16 = 4,
    CK_TILE_HCU_GROUPED_GEMM_INT4 = 5
};

struct ck_tile_hcu_grouped_gemm_desc
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int k_batch;
    int M;
    int N;
    int K;
    int stride_A;
    int stride_B;
    int stride_C;
    int num_d_tensors;
    const void* const* d_ptrs;
    const int* stride_Ds;
};

std::size_t ck_tile_hcu_grouped_gemm_workspace_size(int group_count, int num_d_tensors = 0);

int ck_tile_hcu_grouped_gemm_run(const ck_tile_hcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 int dtype,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream);

} // extern "C"

#endif // CK_GROUPED_GEMM_ABI_DEFINED

// Per-dtype C ABI dispatch wrappers (non-template, defined in instances/)
int grouped_gemm_c_run_fp16(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_fp16_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_bf16(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_bf16_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_gfx936_fp16(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_gfx936_bf16(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_fp8(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_fp8_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_bf8(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_bf8_mpad(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_int8(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);
int grouped_gemm_c_run_int4(const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream);

// Per-dtype example runner wrappers (non-template, defined in instances/)
int grouped_gemm_example_run_fp16(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf16(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_gfx936_fp16(std::string a_layout,
                                         std::string b_layout,
                                         int argc,
                                         char* argv[]);
int grouped_gemm_example_run_gfx936_bf16(std::string a_layout,
                                         std::string b_layout,
                                         int argc,
                                         char* argv[]);
int grouped_gemm_example_run_fp8(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_fp8_128x64(std::string a_layout,
                                        std::string b_layout,
                                        int argc,
                                        char* argv[]);
int grouped_gemm_example_run_fp8_128x128_k32(std::string a_layout,
                                             std::string b_layout,
                                             int argc,
                                             char* argv[]);
int grouped_gemm_example_run_bf8_k64(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf8_128x64(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf8_128x128(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int8_32x32(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int8(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int4(std::string a_layout, std::string b_layout, int argc, char* argv[]);

template <typename GemmConfig>
inline bool grouped_gemm_prefers_large_config(const ck_tile_hcu_grouped_gemm_desc* descs,
                                              int group_count,
                                              int minimum_dimension = 512)
{
    if(descs == nullptr || group_count <= 0)
        return false;

    for(int i = 0; i < group_count; ++i)
    {
        const int kbatch = descs[i].k_batch > 0 ? descs[i].k_batch : 1;
        const int k_alignment = GemmConfig::K_Tile * kbatch;
        if(descs[i].M < minimum_dimension || descs[i].N < minimum_dimension ||
           descs[i].K < minimum_dimension || descs[i].M % GemmConfig::M_Tile != 0 ||
           descs[i].N % GemmConfig::N_Tile != 0 || descs[i].K % k_alignment != 0)
        {
            return false;
        }
    }
    return true;
}

template <typename GemmConfig>
inline bool grouped_gemm_prefers_gfx936_direct_config(
    const ck_tile_hcu_grouped_gemm_desc* descs,
    int group_count,
    int minimum_dimension = 2048)
{
    if(!grouped_gemm_prefers_large_config<GemmConfig>(
           descs, group_count, minimum_dimension))
    {
        return false;
    }
    for(int i = 0; i < group_count; ++i)
    {
        if(descs[i].k_batch != 1 || descs[i].num_d_tensors != 0)
        {
            return false;
        }
    }
    return true;
}

inline std::pair<bool, ck_tile::ArgParser> create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
#if defined(CK_TILE_GROUPED_GEMM_FAST_FP16)
    const char* default_prec = "fp16";
#elif defined(CK_TILE_GROUPED_GEMM_FAST_BF16)
    const char* default_prec = "bf16";
#elif defined(CK_TILE_GROUPED_GEMM_FAST_FP8)
    const char* default_prec = "fp8";
#elif defined(CK_TILE_GROUPED_GEMM_FAST_BF8)
    const char* default_prec = "bf8";
#elif defined(CK_TILE_GROUPED_GEMM_FAST_INT4)
    const char* default_prec = "int4";
#else
    const char* default_prec = "int8";
#endif
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert("stride_As", "", "Tensor A strides - it is empty by default.")
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Cs", "", "Tensor C strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Row by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", default_prec, "data type. fp16/bf16/fp8/bf8/int8/int4")
        .insert("config", "", "optional config selector, e.g. int8_32x32/bf8_128x64/bf8_128x128")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "8", "group count.")
        .insert("kbatch", "1", "kbatch for SplitK")
        .insert("bias", "0", "0: GEMM only, 1: GEMM plus single bias tensor (C=A*B+D)")
        .insert("multiple_d", "0", "0: GEMM only, 1: GEMM plus two D tensors")
        .insert("multiple_d_op", "add", "multi-D epilogue op: add (C+D0+D1) / multiply (C*D0*D1)")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "grouped_gemm.json", "json file name to dump results")
        .insert("debug", "0", "0: no debug prints, 1: print validation summaries")
        .insert("int4_const", "", "optional int4 constant initializer for A/B, e.g. 1")
        .insert("int4_const_a", "", "optional int4 constant initializer for A only")
        .insert("int4_const_b", "", "optional int4 constant initializer for B only")
        .insert("int4_b_one_k", "-1", "debug: set B to one at one logical K and zero elsewhere");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

template <typename GemmConfig>
inline bool grouped_gemm_example_needs_padding(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return false;

    const int group_count = arg_parser.get_int("group_count");
    const int kbatch      = arg_parser.get_int("kbatch");
    const auto Ms         = arg_parser.get_int_vec("Ms");
    const auto Ns         = arg_parser.get_int_vec("Ns");
    const auto Ks         = arg_parser.get_int_vec("Ks");

    if(group_count <= 0 || Ms.size() != static_cast<std::size_t>(group_count) ||
       Ns.size() != static_cast<std::size_t>(group_count) ||
       Ks.size() != static_cast<std::size_t>(group_count))
    {
        // The runner will replace incomplete input vectors with its aligned defaults.
        return false;
    }

    const ck_tile::index_t k_alignment =
        GemmConfig::K_Tile * static_cast<ck_tile::index_t>(kbatch > 0 ? kbatch : 1);
    for(int i = 0; i < group_count; ++i)
    {
        if(Ms[i] % GemmConfig::M_Tile != 0 || Ns[i] % GemmConfig::N_Tile != 0 ||
           Ks[i] % k_alignment != 0)
        {
            return true;
        }
    }
    return false;
}

template <typename GemmConfig>
inline bool grouped_gemm_example_prefers_large_config(int argc,
                                                       char* argv[],
                                                       int minimum_dimension = 512,
                                                       int required_group_count = 0)
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return false;

    const int group_count = arg_parser.get_int("group_count");
    const int kbatch      = arg_parser.get_int("kbatch");
    const auto Ms         = arg_parser.get_int_vec("Ms");
    const auto Ns         = arg_parser.get_int_vec("Ns");
    const auto Ks         = arg_parser.get_int_vec("Ks");

    if(required_group_count > 0 && group_count != required_group_count)
    {
        return false;
    }

    if(group_count <= 0 || Ms.size() != static_cast<std::size_t>(group_count) ||
       Ns.size() != static_cast<std::size_t>(group_count) ||
       Ks.size() != static_cast<std::size_t>(group_count))
    {
        return false;
    }

    const ck_tile::index_t k_alignment =
        GemmConfig::K_Tile * static_cast<ck_tile::index_t>(kbatch > 0 ? kbatch : 1);
    for(int i = 0; i < group_count; ++i)
    {
        if(Ms[i] < minimum_dimension || Ns[i] < minimum_dimension ||
           Ks[i] < minimum_dimension || Ms[i] % GemmConfig::M_Tile != 0 ||
           Ns[i] % GemmConfig::N_Tile != 0 || Ks[i] % k_alignment != 0)
        {
            return false;
        }
    }
    return true;
}

template <typename GemmConfig>
inline bool grouped_gemm_example_prefers_gfx936_direct_config(
    int argc,
    char* argv[],
    int minimum_dimension = 2048)
{
    if(!grouped_gemm_example_prefers_large_config<GemmConfig>(
           argc, argv, minimum_dimension))
    {
        return false;
    }
    auto [result, arg_parser] = create_args(argc, argv);
    return result && arg_parser.get_int("kbatch") == 1 &&
           arg_parser.get_int("bias") == 0 &&
           arg_parser.get_int("multiple_d") == 0;
}

inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg);
}

template <ck_tile::index_t NumDTensor>
inline std::size_t
get_workspace_size(const std::vector<ck_tile::GroupedGemmHostArgsImpl<NumDTensor>>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<NumDTensor>);
}

// // template <typename GemmConfig, typename T>
// // auto shuffle_b(const ck_tile::HostTensor<T>& t)
// // {
// //     assert(t.get_lengths().size() == 2);
// //     int n_ = t.get_lengths()[1];
// //     int k_ = t.get_lengths()[0];

// //     if(ck_tile::is_gfx12_supported())
// //     {
// //         // TODO: Please modify it once kABK0PerLane is changed in WmmaTraitsBase<gfx12>
// //         constexpr int divisor      = 2;
// //         constexpr int kABK0PerLane = 2;
// //         ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
// //                                        GemmConfig::N_Warp_Tile,
// //                                        k_ / GemmConfig::K_Warp_Tile,
// //                                        divisor,
// //                                        kABK0PerLane,
// //                                        GemmConfig::K_Warp_Tile / divisor / kABK0PerLane});
// //         std::copy(t.begin(), t.end(), t_view.begin());
// //         return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
// //     }
// //     else
// //     {
// //         int divisor = 1;
// //         if(ck_tile::is_gfx11_supported())
// //         {
// //             divisor = 1;
// //         }
// //         else
// //         {
// //             assert(is_wave32() == false);
// //             divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
// //         }
// //         ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
// //                                        GemmConfig::N_Warp_Tile,
// //                                        k_ / GemmConfig::K_Warp_Tile,
// //                                        divisor,
// //                                        GemmConfig::K_Warp_Tile / divisor});
// //         std::copy(t.begin(), t.end(), t_view.begin());
// //         return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
// //     }
// // }

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise>
float grouped_gemm(const std::vector<ck_tile::GroupedGemmHostArgsImpl<DsDataType::size()>>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr);

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr,
                            const std::vector<std::pair<void*, std::size_t>>& output_resets,
                            bool splitk = false);
