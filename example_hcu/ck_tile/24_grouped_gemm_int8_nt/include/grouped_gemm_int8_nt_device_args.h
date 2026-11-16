// SPDX-License-Identifier: MIT
#ifndef CK_TILE_EXAMPLE_GROUPED_GEMM_INT8_NT_DEVICE_ARGS_H
#define CK_TILE_EXAMPLE_GROUPED_GEMM_INT8_NT_DEVICE_ARGS_H

#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"

#ifdef __cplusplus
extern "C" {
#endif

// Version 2 adds signed INT8 NT with fused FP32 scales and 16-bit output.
#define CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION 2u

typedef enum ck_tile_hcu_grouped_gemm_int8_flags_v2
{
    // A/B/C strides are K/K/N, A scale is scalar per group, B is per channel,
    // and both scales are present in every nonempty group.
    CK_TILE_HCU_GROUPED_GEMM_INT8_DENSE_CHANNEL_SCALE_V2 = 1u << 0,
    // The caller can bind a packed B view if requested by query.
    CK_TILE_HCU_GROUPED_GEMM_INT8_PACKED_B_AVAILABLE_V2 = 1u << 1
} ck_tile_hcu_grouped_gemm_int8_flags_v2;

typedef enum ck_tile_hcu_grouped_gemm_int8_contract_v2
{
    CK_TILE_HCU_GROUPED_GEMM_INT8_PACKED_B_V2 = 1u << 0
} ck_tile_hcu_grouped_gemm_int8_contract_v2;

typedef struct ck_tile_hcu_grouped_gemm_int8_problem_v2
{
    uint32_t struct_size;
    uint32_t abi_version;
    // CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_{BF16,FP16}_V1.
    int32_t output_type;
    int32_t group_count;
    uint32_t problem_flags; // ck_tile_hcu_grouped_gemm_problem_flags_v1
    uint32_t int8_flags;
    ck_tile_hcu_grouped_gemm_dimension_v1 m;
    int32_t n;
    int32_t k;
    // Upper bound on every actual M. For a common M, this must equal m.extent.
    int32_t max_m;
    // Upper bounds on the group strides, in elements.
    int32_t max_stride_a;
    int32_t max_stride_b;
    int32_t max_stride_c;
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_int8_problem_v2;

// One entry per group in GPU memory. Strides are 0 for scalar and 1 for
// per-row A or per-column B. A null pointer represents unit scale.
typedef struct ck_tile_hcu_grouped_gemm_int8_scale_v2
{
    const float* a;
    const float* b;
    int32_t a_stride;
    int32_t b_stride;
} ck_tile_hcu_grouped_gemm_int8_scale_v2;

typedef struct ck_tile_hcu_grouped_gemm_int8_launch_v2
{
    uint32_t struct_size;
    uint32_t abi_version;
    // GPU GemmTransKernelArg array. A/C are row-major, B is logical NT.
    // Bind raw row-major [N,K] or packed B as specified by query. k_batch = 1.
    const void* device_args;
    const ck_tile_hcu_grouped_gemm_int8_scale_v2* device_scales;
    uint32_t num_cu; // Currently must be zero (default grid).
    uint32_t expected_instance_id;
    void* stream;
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_int8_launch_v2;

typedef struct ck_tile_hcu_grouped_gemm_int8_candidate_info_v2
{
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t instance_id;
    uint32_t architecture_mask;
    uint32_t layout_mask;
    int32_t output_type;
    uint32_t contract_flags;
    char canonical_name[CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NAME_CAPACITY];
    uint64_t reserved[4];
} ck_tile_hcu_grouped_gemm_int8_candidate_info_v2;

// Use the v1 selection record (struct_size and abi_version = 1).
int ck_tile_hcu_grouped_gemm_select_int8_nt_device_args_v2(
    const ck_tile_hcu_grouped_gemm_int8_problem_v2* problem,
    ck_tile_hcu_grouped_gemm_selection_v1* selection);

int ck_tile_hcu_grouped_gemm_query_int8_nt_instance_v2(
    uint32_t instance_id,
    ck_tile_hcu_grouped_gemm_int8_candidate_info_v2* candidate);

int ck_tile_hcu_grouped_gemm_run_int8_nt_device_args_v2(
    const ck_tile_hcu_grouped_gemm_int8_problem_v2* problem,
    const ck_tile_hcu_grouped_gemm_int8_launch_v2* launch,
    ck_tile_hcu_grouped_gemm_selection_v1* selection);

// One entry per group in GPU memory. The caller owns the source and packed B
// storage. Each packed matrix needs N*K bytes and 16-byte alignment.
typedef struct ck_tile_hcu_grouped_gemm_int8_pack_arg_v2
{
    const int8_t* source;
    int8_t* packed;
    int32_t n;
    int32_t k;
    int32_t stride;
} ck_tile_hcu_grouped_gemm_int8_pack_arg_v2;

// Queue weight preprocessing on stream. Keep args and storage alive until
// ready. A consuming stream must be ordered after the preprocessing.
int ck_tile_hcu_grouped_gemm_pack_int8_nt_weights_v2(
    const ck_tile_hcu_grouped_gemm_int8_pack_arg_v2* device_args,
    int32_t group_count,
    int32_t n,
    int32_t k,
    void* stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CK_TILE_EXAMPLE_GROUPED_GEMM_INT8_NT_DEVICE_ARGS_H
