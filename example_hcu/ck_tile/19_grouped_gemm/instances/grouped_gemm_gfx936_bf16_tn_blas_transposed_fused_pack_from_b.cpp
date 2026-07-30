// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Experimental BF16 grad-W entry that computes the byte-equivalent
// column-major transpose and appends packed grad_out^T from its staged B
// operand. The caller supplies [grouped grad-W output | packed grad_out^T].
#include "grouped_gemm_impl.hpp"

extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_fused_pack_from_b_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    if(device_args == nullptr || group_count <= 0)
    {
        return -1;
    }

    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Col,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
                          93608>(ck_tile::stream_config{stream},
                                 group_count,
                                 device_args,
                                 {},
                                 false,
                                 num_cu);
    return 0;
}

extern "C" int
ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_blas_transposed_fused_pack_from_b_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_fused_pack_from_b_device_args(
        device_args, group_count, num_cu, stream);
}
