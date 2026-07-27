// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/conv/block/block_gemm_mmac_nt_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// Adapter from the universal GEMM shape to the HCU ds_read_m32x16 block GEMM
// contract already used by CK Tile convolution. The b16 DS load is exposed as
// an FP16 or BF16 fragment so the following MMAC keeps the input dtype.
template <typename Problem>
struct GemmDsreadmBlockProblem
{
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::CDataType>;
    using Shape       = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockWarps  = typename Shape::BlockWarps;
    using WarpTile    = typename Shape::WarpTile;

    static_assert(is_any_of<ADataType, fp16_t, bf16_t>::value &&
                      std::is_same_v<ADataType, BDataType>,
                  "gfx936 dsreadm grouped GEMM supports matching FP16/BF16 inputs");
    static_assert(WarpTile::at(number<0>{}) == 32 &&
                      WarpTile::at(number<1>{}) == 32 &&
                      WarpTile::at(number<2>{}) == 16,
                  "dsreadm adapter currently requires a 32x32x16 warp tile");

    static constexpr index_t MPerBlock = Shape::kM;
    static constexpr index_t NPerBlock = Shape::kN;
    static constexpr index_t KPerBlock = Shape::kK;
    static constexpr index_t MWarps    = BlockWarps::at(number<0>{});
    static constexpr index_t NWarps    = BlockWarps::at(number<1>{});
    static constexpr index_t MmmacIter = 2;
    static constexpr index_t NmmacIter = 2;
    static constexpr index_t MPermmac  = 16;
    static constexpr index_t NPermmac  = 16;
    static constexpr index_t MmmacInterleave = 1;
    static constexpr index_t NmmacInterleave = 1;
    static constexpr index_t MWarpIter = MPerBlock / (MWarps * 32);
    static constexpr index_t NWarpIter = NPerBlock / (NWarps * 32);
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr bool TransposeC   = Problem::TransposeC;
};

template <typename Problem>
struct GemmDsreadmBlockGemm
    : public BlockGemmMmacNTAsmemBSmemCregV1<GemmDsreadmBlockProblem<Problem>>
{
    using Base = BlockGemmMmacNTAsmemBSmemCregV1<GemmDsreadmBlockProblem<Problem>>;
    using WarpGemm = typename Base::WG;
};

// Variant used only by the staged-operand tuning pipeline below. Instead of
// retaining every K16 operand fragment for a full K64 block, it exposes one
// compile-time K stage at a time. Two such stage buffers can then overlap the
// next ds_read_m32x16 with MMAC on the current operands without changing the
// established full-K DSReadM pipeline.
template <typename Problem>
struct GemmDsreadmStageBlockGemm : public GemmDsreadmBlockGemm<Problem>
{
    using Base = GemmDsreadmBlockGemm<Problem>;
    using WG   = typename Base::WG;
    using ADsreadm =
        WarpDsreadmDispatcher<typename Problem::ADataType,
                              GemmDsreadmBlockProblem<Problem>::MmmacIter,
                              GemmDsreadmBlockProblem<Problem>::MPermmac,
                              GemmDsreadmBlockProblem<Problem>::MmmacInterleave>;
    using BDsreadm =
        WarpDsreadmDispatcher<typename Problem::BDataType,
                              GemmDsreadmBlockProblem<Problem>::NmmacIter,
                              GemmDsreadmBlockProblem<Problem>::NPermmac,
                              GemmDsreadmBlockProblem<Problem>::NmmacInterleave>;

    static constexpr index_t MWarps = GemmDsreadmBlockProblem<Problem>::MWarps;
    static constexpr index_t NWarps = GemmDsreadmBlockProblem<Problem>::NWarps;
    static constexpr index_t MWarpIter =
        GemmDsreadmBlockProblem<Problem>::MWarpIter;
    static constexpr index_t NWarpIter =
        GemmDsreadmBlockProblem<Problem>::NWarpIter;
    static constexpr index_t KWarpIter = Base::KWarpIter;
    static constexpr index_t MPerBlockPerIter = Base::MPerBlockPerIter;
    static constexpr index_t NPerBlockPerIter = Base::NPerBlockPerIter;

    template <index_t KIter, typename ALdsWindow>
    CK_TILE_DEVICE auto LdsLoadAStage(const ALdsWindow& a_lds_window) const
    {
        static_assert(KIter < KWarpIter, "invalid staged A operand index");
        const index_t warp_id_m = ck_tile::get_warp_id() / NWarps;
        auto warp_window        = make_tile_window(
            a_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_lds_window.get_window_origin() +
                multi_index<2>{warp_id_m * WG::kM, KIter * WG::kK},
            make_static_tile_distribution(typename ADsreadm::WarpLoadDstrEncoding{}));

        statically_indexed_array<
            decltype(load_tile_by_dsreadm(ADsreadm{}, warp_window)),
            MWarpIter>
            warp_tensors;
        static_for<0, MWarpIter, 1>{}([&](auto m_warp_iter) {
            warp_tensors(m_warp_iter) =
                load_tile_by_dsreadm(ADsreadm{}, warp_window);
            if constexpr(m_warp_iter != MWarpIter - 1)
            {
                move_tile_window(warp_window,
                                 {MPerBlockPerIter, 0});
            }
        });
        return warp_tensors;
    }

    template <index_t KIter, typename BLdsWindow>
    CK_TILE_DEVICE auto LdsLoadBStage(const BLdsWindow& b_lds_window) const
    {
        static_assert(KIter < KWarpIter, "invalid staged B operand index");
        const index_t warp_id_n = ck_tile::get_warp_id() % NWarps;
        auto warp_window        = make_tile_window(
            b_lds_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_lds_window.get_window_origin() +
                multi_index<2>{warp_id_n * WG::kN, KIter * WG::kK},
            make_static_tile_distribution(typename BDsreadm::WarpLoadDstrEncoding{}));

        statically_indexed_array<
            decltype(load_tile_by_dsreadm(BDsreadm{}, warp_window)),
            NWarpIter>
            warp_tensors;
        static_for<0, NWarpIter, 1>{}([&](auto n_warp_iter) {
            warp_tensors(n_warp_iter) =
                load_tile_by_dsreadm(BDsreadm{}, warp_window);
            if constexpr(n_warp_iter != NWarpIter - 1)
            {
                move_tile_window(warp_window,
                                 {NPerBlockPerIter, 0});
            }
        });
        return warp_tensors;
    }

    template <typename CBlockTensor, typename AWarpTensors, typename BWarpTensors>
    CK_TILE_DEVICE void RunStage(CBlockTensor& c_block_tensor,
                                 const AWarpTensors& a_warp_tensors,
                                 const BWarpTensors& b_warp_tensors) const
    {
        using SFC = space_filling_curve<sequence<MWarpIter, NWarpIter>,
                                        sequence<0, 1>,
                                        sequence<1, 1>>;
        constexpr auto num_access = SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto access_id) {
            constexpr auto idx         = SFC::get_index(access_id);
            constexpr auto m_warp_iter = idx.at(number<0>{});
            constexpr auto n_warp_iter = idx.at(number<1>{});
            using CWarpDstr             = typename WG::CWarpDstr;
            using CWarpTensor           = typename WG::CWarpTensor;
            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros =
                uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            CWarpTensor c_warp_tensor;
            c_warp_tensor.get_thread_buffer() =
                c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                    c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
            WG{}(c_warp_tensor,
                 a_warp_tensors[m_warp_iter],
                 b_warp_tensors[n_warp_iter]);
            c_block_tensor.set_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{},
                                c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                c_warp_tensor.get_thread_buffer());
        });
    }
};

struct GemmPipelineAgBgCrCompV3DsreadmPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV3DsreadmPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return GemmDsreadmBlockGemm<Problem>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t M = Problem::BlockGemmShape::kM;
        constexpr index_t K = Problem::BlockGemmShape::kK;
        constexpr auto packed_k_m = make_naive_tensor_descriptor_packed(
            make_tuple(number<K>{}, number<M>{}), number<8>{});
        return transform_tensor_descriptor(
            packed_k_m,
            make_tuple(make_pass_through_transform(number<K>{}),
                       make_pass_through_transform(number<M>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t N = Problem::BlockGemmShape::kN;
        constexpr index_t K = Problem::BlockGemmShape::kK;
        constexpr auto packed_k_n = make_naive_tensor_descriptor_packed(
            make_tuple(number<K>{}, number<N>{}), number<8>{});
        return transform_tensor_descriptor(
            packed_k_n,
            make_tuple(make_pass_through_transform(number<K>{}),
                       make_pass_through_transform(number<N>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr auto desc = MakeALdsBlockDescriptor<Problem>();
        return integer_least_multiple(
            sizeof(typename Problem::ADataType) * desc.get_element_space_size(), 16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr auto desc = MakeBLdsBlockDescriptor<Problem>();
        return integer_least_multiple(
            sizeof(typename Problem::BDataType) * desc.get_element_space_size(), 16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }
};

// Compute-V3 scheduling with the convolution dsreadm block GEMM as the LDS
// consumer. The production V3/V4 classes are left unchanged.
template <typename Problem, bool StageOperandOverlap = false>
struct GemmPipelineAgBgCrCompV3Dsreadm : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using Policy = GemmPipelineAgBgCrCompV3DsreadmPolicy;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BlockGemm = std::conditional_t<
        StageOperandOverlap,
        GemmDsreadmStageBlockGemm<Problem>,
        remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>>;
    using DistributionBlockGemm = remove_cvref_t<
        decltype(GemmPipelineAgBgCrCompV4DefaultPolicy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    using Base::PrefetchStages;
    using Base::UsePersistentKernel;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }

    static constexpr index_t GetVectorSizeC()
    {
        return Policy::template GetVectorSizeC<Problem>();
    }

    static constexpr index_t GetSmemPackA()
    {
        return Policy::template GetSmemPackA<Problem>();
    }

    static constexpr index_t GetSmemPackB()
    {
        return Policy::template GetSmemPackB<Problem>();
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        return StageOperandOverlap ? "COMPUTE_V3_DSREADM_STAGE"
                                   : "COMPUTE_V3_DSREADM";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrCompV3Dsreadm",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      Problem::kBlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename CBlockTile, typename ALdsWindow, typename BLdsWindow>
    CK_TILE_DEVICE void RunOperandStages(CBlockTile& c_block_tile,
                                         const ALdsWindow& a_lds_gemm_window,
                                         const BLdsWindow& b_lds_gemm_window) const
    {
        static_assert(StageOperandOverlap,
                      "staged operands are only valid for the isolated pipeline");
        static_assert(BlockGemm::KWarpIter == 4,
                      "staged DSReadM currently requires a K64/K16 block");
        const auto block_gemm = BlockGemm{};

        auto a_even =
            block_gemm.template LdsLoadAStage<0>(a_lds_gemm_window);
        auto b_even =
            block_gemm.template LdsLoadBStage<0>(b_lds_gemm_window);
        auto a_odd =
            block_gemm.template LdsLoadAStage<1>(a_lds_gemm_window);
        auto b_odd =
            block_gemm.template LdsLoadBStage<1>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_even, b_even);
        a_even = block_gemm.template LdsLoadAStage<2>(a_lds_gemm_window);
        b_even = block_gemm.template LdsLoadBStage<2>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_odd, b_odd);
        a_odd = block_gemm.template LdsLoadAStage<3>(a_lds_gemm_window);
        b_odd = block_gemm.template LdsLoadBStage<3>(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        block_gemm.RunStage(c_block_tile, a_even, b_even);
        block_gemm.RunStage(c_block_tile, a_odd, b_odd);
    }

    template <bool HasHotLoop,
              TailNumber TailNum,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto RunStageOverlap(
        const ADramBlockWindow& a_dram_block_window,
        const BDramBlockWindow& b_dram_block_window,
        index_t num_loop,
        void* p_smem) const
    {
        static_assert(StageOperandOverlap);
        static_assert(!Problem::DoubleSmemBuffer,
                      "staged Compute V3 DSReadM requires one LDS buffer");

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);
        constexpr auto a_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeBBlockDistributionEncode());
        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window,
                                           a_lds_block,
                                           a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window,
                                           b_lds_block,
                                           b_lds_dstr);
        auto& a_dram_window      = a_windows.get(I0{});
        auto& a_lds_store_window = a_windows.get(I1{});
        auto& a_lds_gemm_window  = a_windows.get(number<2>{});
        auto& b_dram_window      = b_windows.get(I0{});
        auto& b_lds_store_window = b_windows.get(I1{});
        auto& b_lds_gemm_window  = b_windows.get(number<2>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            is_a_col_major ? make_array(KPerBlock, 0)
                           : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_step =
            is_b_row_major ? make_array(KPerBlock, 0)
                           : make_array(0, KPerBlock);

        auto a_reg_tile = load_tile(a_dram_window);
        auto b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        const auto store_a = [&](const auto& src) {
            if constexpr(is_a_col_major)
            {
                auto shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, src);
            }
        };
        const auto store_b = [&](const auto& src) {
            if constexpr(is_b_row_major)
            {
                auto shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, src);
            }
        };

        auto c_block_tile = BlockGemm{}.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        store_a(a_reg_tile);
        store_b(b_reg_tile);
        a_reg_tile = load_tile(a_dram_window);
        b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);
        block_sync_lds();

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                RunOperandStages(c_block_tile,
                                 a_lds_gemm_window,
                                 b_lds_gemm_window);
                block_sync_lds();
                store_a(a_reg_tile);
                store_b(b_reg_tile);
                a_reg_tile = load_tile(a_dram_window);
                b_reg_tile = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);
                block_sync_lds();
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        RunOperandStages(c_block_tile,
                         a_lds_gemm_window,
                         b_lds_gemm_window);
        if constexpr(TailNum == TailNumber::Even)
        {
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            RunOperandStages(c_block_tile,
                             a_lds_gemm_window,
                             b_lds_gemm_window);
        }
        return c_block_tile;
    }

    template <bool HasHotLoop,
              TailNumber TailNum,
              typename ADramBlockWindow,
              typename BDramBlockWindow>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem) const
    {
        static_assert(!Problem::DoubleSmemBuffer,
                      "Compute V3 dsreadm requires one LDS buffer");

        constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        auto&& [a_lds_block, b_lds_block] =
            PipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr auto a_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_dstr = make_static_tile_distribution(
            DistributionBlockGemm::MakeBBlockDistributionEncode());

        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window, a_lds_block, a_lds_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window, b_lds_block, b_lds_dstr);
        auto& a_dram_window      = a_windows.get(I0{});
        auto& a_lds_store_window = a_windows.get(I1{});
        auto& a_lds_gemm_window  = a_windows.get(number<2>{});
        auto& b_dram_window      = b_windows.get(I0{});
        auto& b_lds_store_window = b_windows.get(I1{});
        auto& b_lds_gemm_window  = b_windows.get(number<2>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
        constexpr BDramTileWindowStep b_step =
            is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        auto a_reg_tile = load_tile(a_dram_window);
        auto b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        const auto store_a = [&](const auto& src) {
            if constexpr(is_a_col_major)
            {
                auto shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(a_lds_store_window, src);
            }
        };
        const auto store_b = [&](const auto& src) {
            if constexpr(is_b_row_major)
            {
                auto shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(shuffled, src);
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, shuffled);
            }
            else
            {
                PipelineImplBase{}.LocalPrefill(b_lds_store_window, src);
            }
        };

        auto block_gemm   = BlockGemm{};
        auto c_block_tile = block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        store_a(a_reg_tile);
        store_b(b_reg_tile);

        a_reg_tile = load_tile(a_dram_window);
        b_reg_tile = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);

        block_sync_lds();
        auto a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
        auto b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(HasHotLoop)
        {
            index_t i = 0;
            do
            {
                block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
                block_sync_lds();

                store_a(a_reg_tile);
                store_b(b_reg_tile);

                a_reg_tile = load_tile(a_dram_window);
                b_reg_tile = load_tile(b_dram_window);
                move_tile_window(a_dram_window, a_step);
                move_tile_window(b_dram_window, b_step);

                block_sync_lds();
                a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
                b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
                __builtin_amdgcn_sched_barrier(0);
                ++i;
            } while(i < num_loop - (TailNum == TailNumber::Even ? 2 : 1));
        }

        if constexpr(TailNum == TailNumber::Odd || TailNum == TailNumber::One)
        {
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
        }
        else
        {
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
            block_sync_lds();
            store_a(a_reg_tile);
            store_b(b_reg_tile);
            block_sync_lds();
            a_warp_tiles = block_gemm.LdsLoadA(a_lds_gemm_window);
            b_warp_tiles = block_gemm.LdsLoadB(b_lds_gemm_window);
            block_gemm(c_block_tile, a_warp_tiles, b_warp_tiles);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto run = [&](auto hot_loop, auto tail) {
            if constexpr(StageOperandOverlap)
            {
                return RunStageOverlap<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
            else
            {
                return Run<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem) const
    {
        const auto run = [&](auto hot_loop, auto tail) {
            if constexpr(StageOperandOverlap)
            {
                return RunStageOverlap<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
            else
            {
                return Run<hot_loop.value, tail.value>(
                    a_dram_block_window,
                    b_dram_block_window,
                    num_loop,
                    p_smem);
            }
        };
        return Base::TailHandler(run, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
