// SPDX-License-Identifier: MIT
#pragma once

#include "../include/ck_tile_int8_nt_cached.h"

namespace ck_tile {

namespace int8_routes {
// Route 0 resolves its launch kind through the legacy instance table.
constexpr auto LegacyRaw = CK_TILE_INT8_CACHED_LEGACY_RAW_V1;

enum class Pipeline
{
    Unknown,
    Cooperative,
    Wasp,
    KParallel,
};

struct RouteInfo
{
    Pipeline pipeline;
};

#define CK_INT8_ROUTE(id, abi, kind, ...) constexpr auto id = CK_TILE_INT8_CACHED_##abi##_V1;
#include "grouped_int8_nt_routes.inc"
#undef CK_INT8_ROUTE

constexpr RouteInfo route_info(unsigned route)
{
    switch(route)
    {
#define CK_INT8_PIPELINE_WASP Pipeline::Wasp
#define CK_INT8_PIPELINE_SCALED_WASP Pipeline::Wasp
#define CK_INT8_PIPELINE_PAIRED_WASP Pipeline::Wasp
#define CK_INT8_PIPELINE_COOPERATIVE Pipeline::Cooperative
#define CK_INT8_PIPELINE_K_PARALLEL Pipeline::KParallel
#define CK_INT8_ROUTE(id, abi, kind, ...) \
    case id: return {CK_INT8_PIPELINE_##kind};
#include "grouped_int8_nt_routes.inc"
#undef CK_INT8_ROUTE
#undef CK_INT8_PIPELINE_K_PARALLEL
#undef CK_INT8_PIPELINE_COOPERATIVE
#undef CK_INT8_PIPELINE_PAIRED_WASP
#undef CK_INT8_PIPELINE_SCALED_WASP
#undef CK_INT8_PIPELINE_WASP
    default: return {Pipeline::Unknown};
    }
}

} // namespace int8_routes

namespace int8_instance_ids {
// The key portion of the published ID also identifies the provider entry.
constexpr unsigned RouteBit     = 0x00010000;
constexpr unsigned TwoRowsBit   = 0x00020000;
constexpr unsigned FourRowsBit  = 0x00040000;
constexpr unsigned EightRowsBit = 0x00080000;
constexpr unsigned RowsMask     = TwoRowsBit | FourRowsBit | EightRowsBit;
constexpr unsigned Mask         = 0x0000ffff | RouteBit | RowsMask;

constexpr unsigned from_route(unsigned route, int m = 1)
{
    unsigned row_flag = 0;
    if(route == int8_routes::KParallelDecode)
    {
        if(m == 2)
            row_flag = TwoRowsBit;
        else if(m == 4)
            row_flag = FourRowsBit;
        else if(m == 8)
            row_flag = EightRowsBit;
    }
    return RouteBit | route | row_flag;
}

constexpr unsigned route_for(unsigned instance)
{
    return (instance & RouteBit) ? instance & 0x0000ffff : int8_routes::LegacyRaw;
}
} // namespace int8_instance_ids

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
} // namespace int8_weight_formats

namespace int8_output_types {
    constexpr unsigned Bf16  = CK_TILE_INT8_OUTPUT_BF16_V2;
    constexpr unsigned Fp16  = CK_TILE_INT8_OUTPUT_FP16_V2;
} // namespace int8_output_types

} // namespace ck_tile
