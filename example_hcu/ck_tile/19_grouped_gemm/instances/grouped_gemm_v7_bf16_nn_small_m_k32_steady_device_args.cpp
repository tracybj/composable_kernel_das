// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// P6f independent experimental instance 5014; attribute contract in selector.
#include "grouped_gemm_v7_steady_launch.hpp"
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_k32_steady_device_args(
    void* args, int groups, std::uint32_t num_cu, hipStream_t stream,
    std::uint64_t tiles, ck_tile::index_t tiles_per_group)
{
    (void)num_cu;
    if(args == nullptr || groups <= 0 || groups > ck_tile::kGroupedGemmV7NnMaxGroups || tiles == 0)
        return -1;
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return v7_steady_launch::launch<GemmConfigV7Steady<1>,Row,Row,Row,ck_tile::bf16_t,ck_tile::bf16_t,float,ck_tile::bf16_t>(args,groups,tiles,tiles_per_group,stream);
}
