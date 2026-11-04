// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Experimental instance 5009: 128x128x16, four waves, four LDS stages.
// Common nonragged/nonempty M<=256, G<=1024, K<=4096.
// K>=64, K%16==0. N%128==0; no column tail stores.
#include "grouped_gemm_v7_small_experiment_launch.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_square128_device_args(
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
    return v7_small_experiment_launch::launch<GemmConfigV7SmallExperiment<2>,
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
