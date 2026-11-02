// grouped_gemm_v7_bf16_nn_m01_8_device_args.cpp — CK grouped-GEMM instance entry
// (InstanceId 5002): the InstanceId 5000 NN kernel with the M01 = 8 tile walk.
//
// Same pipeline, epilogue, group lookup, grid and contract as
// grouped_gemm_v7_bf16_nn_device_args.cpp; the only difference is the order in
// which a group's (m, n) tiles are handed to the block ids (GemmConfigV7NnBf16M01_8,
// see grouped_gemm_v7_uniform_launch.hpp). The selector takes it for the K extents
// where the row-major walk loses L2 reuse (K > kGroupedGemmV7NnM01_8KThreshold);
// everything else stays on 5000.
//
// Built in the same archive and under the same v7 codegen contract as 5000 (see
// CMakeLists.txt), and for the same reason kept out of the provider manifest.
//
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "grouped_gemm_v7_uniform_launch.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_m01_8_device_args(
    void* device_args,
    int group_count,
    std::uint32_t num_cu,
    hipStream_t stream,
    std::uint64_t tile_slots,
    ck_tile::index_t tiles_per_group) {
    (void)num_cu;   // the grid is sized by tile_slots, not by the CU count
    if (device_args == nullptr || group_count <= 0 ||
        group_count > ck_tile::kGroupedGemmV7NnMaxGroups || tile_slots == 0) {
        return -1;
    }

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return v7_uniform_launch::launch<GemmConfigV7NnBf16M01_8,
                              Row,
                              Row,
                              Row,
                              ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              float,
                              ck_tile::bf16_t>(device_args,
                                               group_count,
                                               tile_slots,
                                               tiles_per_group,
                                               stream);
}
