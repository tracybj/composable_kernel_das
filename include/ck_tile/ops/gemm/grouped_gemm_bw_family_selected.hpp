// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>

// Launch the selected BW-family FP16/BF16 grouped-GEMM kernels from an
// existing device array of ck_tile::GemmTransKernelArg. The validated family
// currently covers gfx936 and gfx938 under the same aligned-shape contract.
#define CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(suffix)                        \
    extern "C" int ck_tile_hcu_grouped_gemm_bw_family_##suffix##_device_args(       \
        void* device_args,                                                           \
        int group_count,                                                             \
        std::uint32_t num_cu,                                                        \
        hipStream_t stream);                                                         \
    extern "C" int ck_tile_hcu_grouped_gemm_gfx936_##suffix##_device_args(           \
        void* device_args,                                                           \
        int group_count,                                                             \
        std::uint32_t num_cu,                                                        \
        hipStream_t stream)

CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16_nn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16_tn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(fp16_tn_blas_transposed);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_nn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_nn_rr_blas);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_nn_cc_blas_transposed);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_tn);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(bf16_tn_blas_transposed);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(
    bf16_tn_blas_transposed_fused_pack);
CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY(
    bf16_tn_blas_transposed_fused_pack_from_b);

#undef CK_TILE_DECLARE_BW_FAMILY_GROUPED_GEMM_ENTRY
