// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

// BF16 grad-X is exposed as row-major C = A * B. This selected instance
// consumes descriptors rewritten to the equivalent column-major
// C^T = B^T * A^T view, matching the standalone Phase190 C,C -> C winner
// without changing the public pointer or output-storage ABI. C,C -> C is a
// unique integrated kernel type, so keep it untagged: the otherwise useful
// tag wrapper changes BW-family entry scheduling and adds static instructions.
#include "grouped_gemm_impl.hpp"

extern "C" int
ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_cc_blas_transposed_device_args(
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
    using Col    = ck_tile::tensor_layout::gemm::ColumnMajor;

    grouped_gemm_tileloop<Config,
                          Col,
                          Col,
                          Col,
                          ck_tile::bf16_t,
                          ck_tile::bf16_t,
                          float,
                          ck_tile::bf16_t,
                          0>(
        ck_tile::stream_config{stream},
        group_count,
        device_args,
        {},
        false,
        num_cu);
    return 0;
}

extern "C" int
ck_tile_hcu_grouped_gemm_gfx936_bf16_nn_cc_blas_transposed_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream)
{
    return ck_tile_hcu_grouped_gemm_bw_family_bf16_nn_cc_blas_transposed_device_args(
        device_args, group_count, num_cu, stream);
}
