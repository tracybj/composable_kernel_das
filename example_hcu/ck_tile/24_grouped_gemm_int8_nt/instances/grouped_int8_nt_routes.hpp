// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include "../include/ck_tile_int8_nt_cached.h"

namespace ck_tile {

// Formal typed alias for INT8 NT cached grouped GEMM dispatch routes.
// Directly backed by the public C enum ck_tile_int8_nt_cached_route_v1 for ABI stability.
using Int8Route = ck_tile_int8_nt_cached_route_v1;

namespace int8_routes {
    // 0: Classic cooperative double-buffered pipeline (raw fallback)
    constexpr auto LegacyRaw                      = CK_TILE_INT8_CACHED_LEGACY_RAW_V1;

    // 1-2: WASP linear raw (unpacked) variants
    constexpr auto LinearRaw                      = CK_TILE_INT8_CACHED_LINEAR_RAW_V1;
    constexpr auto LinearRawRecord1024            = CK_TILE_INT8_CACHED_LINEAR_RAW_RECORD1024_V1;

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

    // 25: Grid-level Split-K with atomic reduction for high-K decode
    constexpr auto KParallelDecode                = CK_TILE_INT8_CACHED_RAW_K_PARALLEL_M1_M2_V1;

    // 26-30: Paired-Store Epilogue variants (128-bit vector double-write for prefill throughput)
    constexpr auto PairedM256N256                 = CK_TILE_INT8_CACHED_PAIRED_M256_N256_V1;
    constexpr auto PairedM256N384                 = CK_TILE_INT8_CACHED_PAIRED_M256_N384_V1;
    constexpr auto PairedM256N384MaskedM4         = CK_TILE_INT8_CACHED_PAIRED_M256_N384_MASKED_M4_V1;
    constexpr auto PairedM256N384MaskedM8         = CK_TILE_INT8_CACHED_PAIRED_M256_N384_MASKED_M8_V1;
    constexpr auto PairedM128N256                 = CK_TILE_INT8_CACHED_PAIRED_M128_N256_V1;
} // namespace int8_routes

// Architectural and empirical scheduling constants (eliminating magic numbers)
namespace int8_tuning_config {
    // Hardware configuration defaults
    constexpr int GFX936_DEFAULT_NUM_CU           = 80;

    // Tile dimension thresholds
    constexpr int TILE_DIM_16                     = 16;
    constexpr int TILE_DIM_32                     = 32;
    constexpr int TILE_DIM_64                     = 64;
    constexpr int TILE_DIM_128                    = 128;
    constexpr int TILE_DIM_256                    = 256;
    constexpr int TILE_DIM_384                    = 384;

    // Scheduling efficiency trade-off percentages (CTA Wave Packing calibration)
    // When older wave count exceeds wider wave count by >= 147%, switching to wide tile is profitable
    constexpr unsigned WAVE_EFFICIENCY_RATIO_PCT      = 147u;
    constexpr unsigned WAVE_HIGH_OCCUPANCY_RATIO_PCT = 144u;
    constexpr unsigned BATCH_EFFICIENCY_RATIO_PCT    = 146u;
    constexpr unsigned HIGH_WAVE_COUNT_THRESHOLD     = 24u;
    constexpr unsigned HIGH_BATCH_COUNT_THRESHOLD    = 20u;
} // namespace int8_tuning_config

// Human-readable route name query helper
inline const char* get_int8_route_name(unsigned route)
{
    switch(route)
    {
    case int8_routes::LegacyRaw:                  return "CooperativeFallback (Route 0)";
    case int8_routes::LinearRaw:                  return "WaspRawM256N256 (Route 1)";
    case int8_routes::LinearRawRecord1024:         return "WaspRawM256N256Record1024 (Route 2)";
    case int8_routes::LinearTransposedRaw:         return "WaspRawTransposeM256N256 (Route 3)";
    case int8_routes::LinearTransposedRecord1024:  return "WaspRawTransposeM256N256Record1024 (Route 4)";
    case int8_routes::PackedWeightM256N256:        return "WaspPackedM256N256 (Route 5)";
    case int8_routes::PackedM128N256:              return "WaspPackedM128N256 (Route 6)";
    case int8_routes::PackedM128N128:              return "WaspPackedM128N128 (Route 7)";
    case int8_routes::PackedM64N256:               return "WaspPackedM64N256 (Route 8)";
    case int8_routes::PackedM64N64:                return "WaspPackedM64N64 (Route 9)";
    case int8_routes::PackedM32N256:               return "WaspPackedM32N256 (Route 10)";
    case int8_routes::PackedM32N64:                return "WaspPackedM32N64 (Route 11)";
    case int8_routes::PackedM256N384:              return "WaspPackedM256N384 (Route 12)";
    case int8_routes::PackedM256N384MaskedM4:      return "WaspPackedM256N384MaskedM4 (Route 22)";
    case int8_routes::PackedM256N384MaskedM8:      return "WaspPackedM256N384MaskedM8 (Route 23)";
    case int8_routes::PackedCooperativeM1N256:     return "PackedCooperativeM1N256 (Route 24)";
    case int8_routes::KParallelDecode:             return "KParallelGridSplitK (Route 25)";
    case int8_routes::PairedM256N256:              return "PairedWaspPackedM256N256 (Route 26)";
    case int8_routes::PairedM256N384:              return "PairedWaspPackedM256N384 (Route 27)";
    case int8_routes::PairedM256N384MaskedM4:      return "PairedWaspMaskedM4 (Route 28)";
    case int8_routes::PairedM256N384MaskedM8:      return "PairedWaspMaskedM8 (Route 29)";
    case int8_routes::PairedM128N256:              return "PairedWaspPackedM128N256 (Route 30)";
    default:                                       return "UnknownRoute";
    }
}

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
