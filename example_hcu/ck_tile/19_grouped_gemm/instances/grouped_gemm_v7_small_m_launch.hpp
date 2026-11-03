// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "grouped_gemm_v7_uniform_launch.hpp"
#include "../grouped_gemm_v7_small_m_epilogue.hpp"
template <ck_tile::index_t TileN>
struct GemmConfigV7SmallM : GemmConfigV7NnBf16
{
    static constexpr ck_tile::index_t M_Tile=64, N_Tile=TileN;
    static constexpr ck_tile::index_t M_Warp=2, N_Warp=2;
    static constexpr int kBlockPerCu=2;
};
namespace v7_small_m_launch {
inline std::uint64_t block_id_range(const ck_tile::GroupedGemmProblem& problem,
                                  std::uint32_t num_cu, ck_tile::index_t tile_n)
{
    const auto tiles = static_cast<std::uint64_t>(ck_tile::grouped_gemm_v7_small_m_tile_bound(problem,tile_n));
    const std::uint64_t slots=num_cu>0?num_cu:1;
    return (tiles+slots-1)/slots*slots;
}
inline ck_tile::index_t tiles_per_group(const ck_tile::GroupedGemmProblem& problem,
                                       ck_tile::index_t tile_n)
{
    if(problem.group_count<=ck_tile::kGroupedGemmV7ScalarScanMaxGroups ||
       problem.m.kind!=ck_tile::GroupedGemmDimensionKind::common || problem.group_lengths_are_ragged)
        return 0;
    return static_cast<ck_tile::index_t>(((problem.m.extent+63)/64)*((problem.n.extent+tile_n-1)/tile_n));
}
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

    using Pipeline = ck_tile::GemmPipelineV7SmallM<Problem, Config::N_Tile>;
    using Epilogue =
        ck_tile::GroupedGemmV7SmallMStoreEpilogue<CDataType, typename Pipeline::AccTile, typename Pipeline::G>;
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

    const auto kernel = ck_tile::make_kernel<256, Config::kBlockPerCu>(
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

} // namespace v7_small_m_launch
