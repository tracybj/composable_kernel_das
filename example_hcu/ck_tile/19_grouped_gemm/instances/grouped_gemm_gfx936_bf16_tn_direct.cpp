// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Keep the BF16 C,R grad-W specialization in its own translation unit. The
// selected BW-family R,C/R,R kernels use a Triton-specific fused epilogue, while
// this hipBLASLt-inspired DSReadM pipeline intentionally uses the same
// non-fused epilogue contract as its independently validated performance gate.
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_CIRCULAR
#define CK_TILE_GROUPED_GEMM_DSREADM_DIRECT_G2L_TN_SCALAR_RESOURCE
#define CK_TILE_GROUPED_GEMM_DSREADM_B_OPERAND_PIPELINED

#include "grouped_gemm_impl.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    if(device_args == nullptr || group_count <= 0)
    {
        return -1;
    }

    (void)num_cu;
    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Row,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t>(ck_tile::stream_config{stream},
                                           group_count,
                                           device_args,
                                           {},
                                           false);
    return 0;
}

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_device_args(
        device_args, group_count, num_cu, stream);
}
