// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Info Technologies Ltd.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mls_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = GemmPipelineAgBgCrMlsDefaultPolicy>
struct GemmPipelineAgBgCrMls
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using AMlsLayout     = ALayout;
    using BMlsLayout     = std::conditional_t<
        std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr index_t NumLdsStages = 2;
    using WarpTile = typename BlockGemmShape::WarpTile;
    static constexpr index_t MmmacInterleave = WarpTile::at(number<0>{}) / 16;
    static constexpr index_t NmmacInterleave = WarpTile::at(number<1>{}) / 16;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool Preshuffle       = false;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;
    static constexpr index_t NumWaveGroups = 1;
    static constexpr bool IsMlsPipeline    = true;

    static_assert((std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> ||
                   std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>) &&
                      (std::is_same_v<BLayout, tensor_layout::gemm::RowMajor> ||
                       std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>),
                  "Grouped-GEMM MLS requires GEMM row-major or column-major inputs");
    static_assert(MPerBlock == 128 && NPerBlock == 128 && KPerBlock == 32 &&
                      BlockSize == 256,
                  "The initial grouped-GEMM MLS pipeline is validated for 128x128x32/256 only");

    using WaspHelper = wasp_helper<BlockSize / get_warp_size(), BlockSize, false>;
    static constexpr auto AMlsWindowLengths = sequence<32, 32>{};
    static constexpr auto BMlsWindowLengths = sequence<32, 32>{};

    template <bool = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }

    template <bool = false>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC() { return 2; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackA() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPackB() { return 1; }
    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC() { return bool_constant<false>{}; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetLdsByteSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsSupported(index_t num_loop)
    {
        return num_loop % NumLdsStages == 0 && num_loop / NumLdsStages > 1;
    }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop / NumLdsStages > 2;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Full;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto
    TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber)
    {
        if(has_hot_loop)
        {
            return run_func(bool_constant<true>{},
                            integral_constant<TailNumber, TailNumber::Full>{});
        }
        return run_func(bool_constant<false>{},
                        integral_constant<TailNumber, TailNumber::Full>{});
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrMls",
                      concat('x', MPerBlock, NPerBlock, KPerBlock, BlockSize),
                      concat('x', NumLdsStages));
    }

    template <typename AView, typename BView>
    CK_TILE_DEVICE auto operator()(const AView& a_view,
                                   const BView& b_view,
                                   index_t block_origin_m,
                                   index_t block_origin_n,
                                   index_t stride_a,
                                   index_t stride_b,
                                   index_t k_remainder,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   void* p_smem) const
    {
        auto a_mls_window =
            make_tile_window_mls(a_view,
                                 sequence<number<MPerBlock>{}, number<KPerBlock>{}>{},
                                 AMlsWindowLengths,
                                 {block_origin_m, 0},
                                 AMlsLayout{},
                                 number<MmmacInterleave>{},
                                 WaspHelper{},
                                 stride_a,
                                 k_remainder);

        auto b_mls_window =
            make_tile_window_mls(b_view,
                                 sequence<number<NPerBlock>{}, number<KPerBlock>{}>{},
                                 BMlsWindowLengths,
                                 {block_origin_n, 0},
                                 BMlsLayout{},
                                 number<NmmacInterleave>{},
                                 WaspHelper{},
                                 stride_b,
                                 k_remainder);

        constexpr auto a_lds_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
        constexpr auto b_lds_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t a_lds_elements =
            Policy::template GetALdsElementSize<Problem>();
        constexpr index_t b_lds_elements =
            Policy::template GetBLdsElementSize<Problem>();

        auto a_lds_windows = lds_utils::AllocateLdsWindows<ADataType, NumLdsStages>(
            p_smem,
            a_lds_elements,
            0,
            KPerBlock,
            make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
            make_multi_index(0, 0),
            a_lds_desc);

        auto b_lds_windows = lds_utils::AllocateLdsWindows<BDataType, NumLdsStages>(
            p_smem,
            b_lds_elements,
            NumLdsStages * a_lds_elements,
            KPerBlock,
            make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
            make_multi_index(0, 0),
            b_lds_desc);

        if(has_hot_loop)
        {
            return Run<true>(
                a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
        }
        return Run<false>(
            a_mls_window, a_lds_windows, b_mls_window, b_lds_windows, num_loop);
    }

    template <bool HasHotLoop,
              typename AMlsWindow,
              typename ALdsWindows,
              typename BMlsWindow,
              typename BLdsWindows>
    CK_TILE_DEVICE auto Run(AMlsWindow& a_mls_window,
                            ALdsWindows& a_lds_windows,
                            BMlsWindow& b_mls_window,
                            BLdsWindows& b_lds_windows,
                            index_t num_loop) const
    {
        const auto block_gemm = Policy::template GetBlockGemm<Problem>();
        auto c_block_tile     = block_gemm.MakeCBlockTile();

        static_for<0, NumLdsStages, 1>{}([&](auto i) {
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr());
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr());
            a_mls_window.advance();
            b_mls_window.advance();
        });

        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
        async_load_fence((a_mls_window.get_num_of_access() +
                          b_mls_window.get_num_of_access()) *
                         (NumLdsStages - 1));
        wg_sync();

        promote_prio();
        auto a_warp_tensors =
            block_gemm.GetAWarpTensors(a_lds_windows(number<0>{}).get_buffer_ptr());
        auto b_warp_tensors =
            block_gemm.GetBWarpTensors(b_lds_windows(number<0>{}).get_buffer_ptr());
        restore_prio();

        const auto main_loop = [&](auto i, auto pad_last_stage) {
            wg_sync_lds(bool_constant<true>{});
            promote_prio();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            restore_prio();
            wg_sync();

            constexpr bool pad_k =
                pad_last_stage.value && (i == NumLdsStages - 1);
            a_mls_window.async_mls_load_asm(a_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            b_mls_window.async_mls_load_asm(b_lds_windows(i).get_buffer_ptr(),
                                            bool_constant<pad_k>{});
            a_mls_window.advance();
            b_mls_window.advance();

            async_load_fence((a_mls_window.get_num_of_access() +
                              b_mls_window.get_num_of_access()) *
                             (NumLdsStages - 1));
            wg_sync();

            promote_prio();
            a_warp_tensors = block_gemm.GetAWarpTensors(
                a_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr());
            b_warp_tensors = block_gemm.GetBWarpTensors(
                b_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr());
            restore_prio();
        };

        const auto tail_loop = [&](auto i) {
            wg_sync_lds(bool_constant<true>{});
            promote_prio();
            block_gemm(c_block_tile, a_warp_tensors, b_warp_tensors);
            restore_prio();
            wg_sync();

            if constexpr(i != NumLdsStages - 1)
            {
                buffer_load_fence(0);
                wg_sync();
                promote_prio();
                a_warp_tensors = block_gemm.GetAWarpTensors(
                    a_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr());
                b_warp_tensors = block_gemm.GetBWarpTensors(
                    b_lds_windows(number<(i + 1) % NumLdsStages>{}).get_buffer_ptr());
                restore_prio();
            }
        };

        if constexpr(HasHotLoop)
        {
            index_t loop = 0;
            do
            {
                static_for<0, NumLdsStages, 1>{}(
                    [&](auto i) { main_loop(i, bool_constant<false>{}); });
                loop += NumLdsStages;
            } while(loop < num_loop - 2 * NumLdsStages);
        }

        static_for<0, NumLdsStages, 1>{}(
            [&](auto i) { main_loop(i, bool_constant<true>{}); });
        static_for<0, NumLdsStages, 1>{}([&](auto i) { tail_loop(i); });

        return c_block_tile;
    }
};

} // namespace ck_tile
