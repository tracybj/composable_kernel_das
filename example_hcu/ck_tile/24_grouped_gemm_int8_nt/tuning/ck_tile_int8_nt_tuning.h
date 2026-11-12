// SPDX-License-Identifier: MIT
#pragma once

#include "../include/ck_tile_int8_nt_cached.h"

#ifdef __cplusplus
extern "C" {
#endif

// The route and legacy_instance pair is stable within the reported build.
// A zero route uses the legacy instance ID; other routes use legacy_instance 0.
typedef struct ck_tile_int8_nt_tuning_candidate_v1
{
    uint32_t route, legacy_instance;
    int32_t bm, bn, bk, threads;
    uint32_t packed_weight;
} ck_tile_int8_nt_tuning_candidate_v1;

// These entry points exist only in CK_INT8_CACHED_TUNING builds. Candidates are
// native executions for uniform nonempty BF16/FP16 groups. Pass candidates=NULL
// and capacity=0 to query count; pass a buffer of at least that size to fill it.
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_tuning_list_v1(
    ck_tile_int8_nt_cached_weights_handle_v1 weights,
    const ck_tile_int8_nt_cached_group_desc_v2* groups,
    int32_t group_count,
    ck_tile_int8_nt_tuning_candidate_v1* candidates,
    size_t capacity,
    size_t* count);

// Validates the candidate against the actual descriptors before binding raw or
// packed weights. The returned plan uses the ordinary query/run/destroy API.
CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_tuning_plan_create_v1(
    ck_tile_int8_nt_cached_weights_handle_v1 weights,
    const ck_tile_int8_nt_cached_group_desc_v2* groups,
    int32_t group_count,
    uint32_t route,
    uint32_t legacy_instance,
    ck_tile_int8_nt_cached_plan_handle_v1* plan);

#ifdef __cplusplus
}
#endif
