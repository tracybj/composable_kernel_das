// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "grouped_gemm_v7_uniform_launch.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_cache_noslc.hpp"
namespace v7_cache_launch {
template <typename Config,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
inline int launch(const void* kargs_dev,
                  int group_count,
                  std::uint64_t tile_slots,
                  ck_tile::index_t tiles_per_group,
                  hipStream_t stream)
{
    using Shape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;

    using Traits = ck_tile::TileGemmUniversalTraits<Config::kPadM,
                                                    Config::kPadN,
                                                    Config::kPadK,
                                                    Config::DoubleSmemBuffer,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    Config::TransposeC,
                                                    false,
                                                    false,
                                                    1,
                                                    false,
                                                    false,
                                                    Config::SupportsFastRowMajorB,
                                                    Config::SupportsFastColumnMajorA>;

    using Problem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                          BDataType,
                                                          AccDataType,
                                                          Shape,
                                                          Traits,
                                                          Config::Scheduler,
                                                          true,
                                                          ck_tile::TailNumber::Full,
                                                          ck_tile::null_type,
                                                          ck_tile::null_type,
                                                          false,
                                                          2,
                                                          2>;

    using Pipeline = ck_tile::GemmPipelineV7CacheNoSlc<Problem, ck_tile::V7UniformKind::NN>;
    using Epilogue =
        ck_tile::GroupedGemmV7UniformStoreEpilogue<CDataType, typename Pipeline::AccTile>;
    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape,
                                                                  Config::TileParitionerGroupNum,
                                                                  Config::TileParitionerM01>;
    using Kernel = ck_tile::GroupedGemmV7UniformKernel<
        ck_tile::GroupedGemmKernel<Partitioner, Pipeline, Epilogue>>;

    // The grid is one CTA per tile slot and the kernel's block ids are index_t.
    if(kargs_dev == nullptr || group_count <= 0 || tile_slots == 0 ||
       tile_slots > static_cast<std::uint64_t>(std::numeric_limits<ck_tile::index_t>::max()) ||
       tiles_per_group < 0)
    {
        return -1;
    }

    const dim3 grid(static_cast<unsigned>(tile_slots));
    const dim3 block(Kernel::BlockSize());

    const auto kernel = ck_tile::make_kernel<512, Config::kBlockPerCu>(
        Kernel{},
        grid,
        block,
        Kernel::GetSmemSize(),
        ck_tile::cast_pointer_to_constant_address_space(kargs_dev),
        group_count,
        tiles_per_group);
    ck_tile::launch_kernel(ck_tile::stream_config{stream}, kernel);
    return 0;
}

}   // namespace v7_cache_launch
