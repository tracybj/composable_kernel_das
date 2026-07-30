// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// FP16 counterpart of the BLAS-compatible BF16 grad-W transposed view.
#include "grouped_gemm_impl.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_bw_family_fp16_tn_blas_transposed_device_args(
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
    using Config = GemmConfigComputeV3DsreadmStage256W8<ck_tile::half_t>;
    using Row    = ck_tile::tensor_layout::gemm::RowMajor;
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

    grouped_gemm_tileloop<Config,
                          Col,
                          Row,
                          Col,
                          ck_tile::half_t,
                          ck_tile::half_t,
                          float,
                          ck_tile::half_t>(ck_tile::stream_config{stream},
                                           group_count,
                                           device_args,
                                           {},
                                           false);
    return 0;
}

extern "C" int ck_tile_hcu_grouped_gemm_gfx936_fp16_tn_blas_transposed_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_fp16_tn_blas_transposed_device_args(
        device_args, group_count, num_cu, stream);
}
