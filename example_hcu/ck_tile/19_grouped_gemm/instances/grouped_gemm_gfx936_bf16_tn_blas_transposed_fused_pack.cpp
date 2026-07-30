// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// Experimental BF16 grad-W entry that appends a packed grad_out^T tensor to
// the output allocation. The caller provides one contiguous allocation:
// [grouped grad-W output | packed grad_out^T].
#include "grouped_gemm_impl.hpp"

extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_fused_pack_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    if(device_args == nullptr || group_count <= 0)
    {
        return -1;
    }

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_M128N256)
    using Config = GemmConfigComputeV3DsreadmStage128x256W8<ck_tile::bf16_t>;
#else
    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::bf16_t>;
#endif
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

#if defined(CK_TILE_GROUPED_GEMM_FUSED_PACK_GRADX_ROW_MAJOR_C)
    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Row,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
                          93607>(ck_tile::stream_config{stream},
                                 group_count,
                                 device_args,
                                 {},
                                 false,
                                 num_cu);
#else
    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Col,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
                          93607>(ck_tile::stream_config{stream},
                                           group_count,
                                           device_args,
                                           {},
                                           false,
                                           num_cu);
#endif
    return 0;
}

extern "C" int
ck_tile_hcu_grouped_gemm_gfx936_bf16_tn_blas_transposed_fused_pack_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_bf16_tn_blas_transposed_fused_pack_device_args(
        device_args, group_count, num_cu, stream);
}
