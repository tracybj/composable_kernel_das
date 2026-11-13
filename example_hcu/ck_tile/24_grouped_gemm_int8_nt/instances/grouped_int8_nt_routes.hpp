// SPDX-License-Identifier: MIT
#pragma once

#include "../include/ck_tile_int8_nt_cached.h"

namespace ck_tile {

namespace int8_routes {
    // 0: Classic cooperative double-buffered pipeline (raw fallback)
    constexpr auto LegacyRaw                      = CK_TILE_INT8_CACHED_LEGACY_RAW_V1;

    // 3-4: WASP transposed raw variants
    constexpr auto LinearTransposedRaw            = CK_TILE_INT8_CACHED_LINEAR_TRANSPOSED_RAW_V1;
    constexpr auto LinearTransposedRecord1024     = CK_TILE_INT8_CACHED_LINEAR_TRANSPOSED_RAW_RECORD1024_V1;

    // 5-12: WASP packed weight variants with standard epilogue
    constexpr auto PackedWeightM256N256           = CK_TILE_INT8_CACHED_PACKED_WEIGHT_V1;
    constexpr auto PackedM128N256                 = CK_TILE_INT8_CACHED_PACKED_M128_N256_A2_V1;
    constexpr auto PackedM128N128                 = CK_TILE_INT8_CACHED_PACKED_M128_N128_A4_V1;
    constexpr auto PackedM64N256                  = CK_TILE_INT8_CACHED_PACKED_M64_N256_A2_V1;
    constexpr auto PackedM64N64                   = CK_TILE_INT8_CACHED_PACKED_M64_N64_A2_V1;
    constexpr auto PackedM32N256                  = CK_TILE_INT8_CACHED_PACKED_M32_N256_A2_V1;
    constexpr auto PackedM32N64                   = CK_TILE_INT8_CACHED_PACKED_M32_N64_A1_V1;
    constexpr auto PackedM256N384                 = CK_TILE_INT8_CACHED_PACKED_M256_N384_A2_V1;

    // 22-23: Tail shape masked wide-N variants (standard epilogue)
    constexpr auto PackedM256N384MaskedM4         = CK_TILE_INT8_CACHED_PACKED_M256_N384_MASKED_M4_V1;
    constexpr auto PackedM256N384MaskedM8         = CK_TILE_INT8_CACHED_PACKED_M256_N384_MASKED_M8_V1;

    // 24: Packed-Cooperative zero-LDS bypass for ultra-narrow decode (M <= 8)
    constexpr auto PackedCooperativeM1N256        = CK_TILE_INT8_CACHED_PACKED_M16_N64_K256_V1;

    // 25: Wave partial sums with in-CTA reduction for long-K decode
    constexpr auto KParallelDecode                = CK_TILE_INT8_CACHED_RAW_K_PARALLEL_M1_M2_V1;

    // 26-30: Paired-Store Epilogue variants (128-bit vector double-write for prefill throughput)
    constexpr auto PairedM256N256                 = CK_TILE_INT8_CACHED_PAIRED_M256_N256_V1;
    constexpr auto PairedM256N384                 = CK_TILE_INT8_CACHED_PAIRED_M256_N384_V1;
    constexpr auto PairedM256N384MaskedM4         = CK_TILE_INT8_CACHED_PAIRED_M256_N384_MASKED_M4_V1;
    constexpr auto PairedM256N384MaskedM8         = CK_TILE_INT8_CACHED_PAIRED_M256_N384_MASKED_M8_V1;
    constexpr auto PairedM128N256                 = CK_TILE_INT8_CACHED_PAIRED_M128_N256_V1;

    // 31-32: Scaled raw-weight cooperative decode instances.
    constexpr auto RawAlignedM16N32K512          = CK_TILE_INT8_CACHED_RAW_ALIGNED_M16_N32_K512_V1;
    constexpr auto RawM32N64K256W4               = CK_TILE_INT8_CACHED_RAW_M32_N64_K256_W4_V1;

    constexpr bool uses_packed_weights(unsigned route)
    {
        switch(route)
        {
        case PackedWeightM256N256:
        case PackedM128N256:
        case PackedM128N128:
        case PackedM64N256:
        case PackedM64N64:
        case PackedM32N256:
        case PackedM32N64:
        case PackedM256N384:
        case PackedM256N384MaskedM4:
        case PackedM256N384MaskedM8:
        case PackedCooperativeM1N256:
        case PairedM256N256:
        case PairedM256N384:
        case PairedM256N384MaskedM4:
        case PairedM256N384MaskedM8:
        case PairedM128N256: return true;
        default: return false;
        }
    }
} // namespace int8_routes

// Architectural and empirical scheduling constants (eliminating magic numbers)
namespace int8_tuning_config {
    // Hardware configuration defaults
    constexpr int GFX936_DEFAULT_NUM_CU           = 80;

    // Scheduling efficiency trade-off percentages (CTA Wave Packing calibration)
    // When older wave count exceeds wider wave count by >= 147%, switching to wide tile is profitable
    constexpr unsigned WAVE_EFFICIENCY_RATIO_PCT      = 147u;
    constexpr unsigned WAVE_HIGH_OCCUPANCY_RATIO_PCT = 144u;
    constexpr unsigned BATCH_EFFICIENCY_RATIO_PCT    = 146u;
    constexpr unsigned HIGH_WAVE_COUNT_THRESHOLD     = 24u;
    constexpr unsigned HIGH_BATCH_COUNT_THRESHOLD    = 20u;
} // namespace int8_tuning_config

namespace int8_weight_formats {
    constexpr int Raw    = 0;
    constexpr int Packed = 1;
} // namespace int8_weight_formats

namespace int8_output_types {
    constexpr unsigned Int32 = CK_TILE_INT8_OUTPUT_INT32_V2;
    constexpr unsigned Bf16  = CK_TILE_INT8_OUTPUT_BF16_V2;
    constexpr unsigned Fp16  = CK_TILE_INT8_OUTPUT_FP16_V2;
} // namespace int8_output_types

namespace int8_scale_modes {
    constexpr unsigned Scalar      = CK_TILE_INT8_SCALE_SCALAR_V2;
    constexpr unsigned OuterVector = CK_TILE_INT8_SCALE_OUTER_VECTOR_V2;
} // namespace int8_scale_modes

} // namespace ck_tile
