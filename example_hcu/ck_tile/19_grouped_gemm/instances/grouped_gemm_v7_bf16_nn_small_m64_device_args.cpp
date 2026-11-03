// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// Experimental instance 5004: 64x64x16, four waves, four stages.
// Selector contract: bf16 NN gfx936, common non-ragged M<=256, 64<=K<=4096,
// K%16==0, N%64==0, nonempty groups. No column tail stores.
#include "grouped_gemm_v7_small_m_launch.hpp"

extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m64_device_args(
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
    return v7_small_m_launch::launch<GemmConfigV7SmallM<64>,
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
