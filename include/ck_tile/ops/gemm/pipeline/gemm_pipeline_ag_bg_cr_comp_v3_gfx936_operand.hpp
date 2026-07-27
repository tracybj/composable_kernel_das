// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"

namespace ck_tile {

// gfx936 aligned RC/NN/TN specialization for a 256x256x64/W8 lowering.
// One K64 global tile is prefetched in registers while the current K64 tile
// resides in LDS. Unlike generic V3, MMAC operands are loaded from LDS one K16
// slice at a time, so the full replicated A/B block tile never coexists with
// the 128-VGPR accumulator. The custom block policy updates C vectors in place.
template <typename Problem,
          typename Policy = GemmPipelineAgBgCrCompV3TritonOperandPolicy>
struct GemmPipelineAgBgCrCompV3TritonOperand
    : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;
    static constexpr index_t KPerStage = BlockGemmShape::WarpTile::at(number<2>{});
    static constexpr index_t NumKStages = KPerBlock / KPerStage;
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    using StageBlockGemmShape =
        TileGemmShape<sequence<MPerBlock, NPerBlock, KPerStage>,
                      typename BlockGemmShape::BlockWarps,
                      typename BlockGemmShape::WarpTile,
                      BlockGemmShape::PermuteA,
                      BlockGemmShape::PermuteB>;

    struct StageProblem : Problem
    {
        using BlockGemmShape = StageBlockGemmShape;
    };

    using FullBlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using StageBlockGemm =
        remove_cvref_t<decltype(Policy::template GetBlockGemm<StageProblem>())>;

    static constexpr bool IsARow =
        std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>;
    static constexpr bool IsBRow =
        std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
    static_assert(is_any_of<ALayout,
                            tensor_layout::gemm::RowMajor,
                            tensor_layout::gemm::ColumnMajor>::value &&
                      is_any_of<BLayout,
                                tensor_layout::gemm::RowMajor,
                                tensor_layout::gemm::ColumnMajor>::value,
                  "gfx936 operand pipeline requires a standard GEMM layout");
    static_assert(MPerBlock == 256 && NPerBlock == 256 && KPerBlock == 64 &&
                      KPerStage == 16 && NumKStages == 4 &&
                      BlockGemmShape::WarpTile::at(number<0>{}) == 16 &&
                      BlockGemmShape::WarpTile::at(number<1>{}) == 16,
                  "Triton operand pipeline requires a 256x256x64 block with "
                  "eight interleaved M repeats and four interleaved N repeats "
                  "of the native M16xN16xK16 MMAC tile per wave");
    static_assert(BlockGemmShape::NumWarps == 8,
                  "Triton operand pipeline requires eight waves");
    static_assert(!Problem::kPadM && !Problem::kPadN && !Problem::kPadK &&
                      !Problem::DoubleSmemBuffer,
                  "Triton operand pipeline requires aligned single-LDS-buffer input");

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
        return "COMPUTE_V3_TRITON_OPERAND";
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_',
                      "pipeline_AgBgCrCompV3TritonOperand",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <index_t Offset0, index_t Offset1, typename Vec4>
    CK_TILE_DEVICE static void StoreLdsPairSt64(index_t byte_address,
                                                const Vec4& x0,
                                                const Vec4& x1)
    {
        static_assert(sizeof(Vec4) == 8,
                      "ds_write2st64_b64 requires two 64-bit payloads");
        asm volatile("ds_write2st64_b64 %0, %1, %2 offset0:%3 offset1:%4"
                     :
                     : "v"(byte_address),
                       "v"(bit_cast<fp32x2_t>(x0)),
                       "v"(bit_cast<fp32x2_t>(x1)),
                       "n"(Offset0),
                       "n"(Offset1)
                     : "memory");
    }

    template <typename Vec4>
    CK_TILE_DEVICE static void StoreLdsB64(index_t byte_address, const Vec4& x)
    {
        static_assert(sizeof(Vec4) == 8, "ds_write_b64 requires a 64-bit payload");
        asm volatile("ds_write_b64 %0, %1"
                     :
                     : "v"(byte_address), "v"(bit_cast<fp32x2_t>(x))
                     : "memory");
    }

    template <index_t Offset0, index_t Offset1>
    CK_TILE_DEVICE static fp32x4_t LoadLdsPairSt64(index_t byte_address)
    {
        fp32x4_t payload;
        asm volatile("ds_read2st64_b64 %0, %1 offset0:%2 offset1:%3"
                     : "=v"(payload)
                     : "v"(byte_address), "n"(Offset0), "n"(Offset1)
                     : "memory");
        return payload;
    }

    CK_TILE_DEVICE static void WaitLdsReads()
    {
        // Inline DS reads are opaque to the compiler's waitcnt dependency
        // tracker. Do not let the following MMAC consume their destination
        // VGPRs before lgkmcnt reaches zero. Once correctness is established,
        // this full wait is split and interleaved with MMAC like Triton.
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
    }

    template <index_t Count>
    CK_TILE_DEVICE static void WaitLdsReads(number<Count>)
    {
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
        asm volatile("s_waitcnt lgkmcnt(%0)"
                     :
                     : "n"(Count)
                     : "memory");
#if !defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_NO_SCHED_BARRIER)
        __builtin_amdgcn_sched_barrier(0);
#endif
    }

    template <typename ADramBlockWindow,
              typename BDramBlockWindow,
              typename FinalEpiloguePreStore>
    CK_TILE_DEVICE auto Run(const ADramBlockWindow& a_dram_block_window,
                            const BDramBlockWindow& b_dram_block_window,
                            index_t num_loop,
                            void* p_smem,
                            [[maybe_unused]]
                            FinalEpiloguePreStore&& final_epilogue_pre_store) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindow::DataType>> &&
                std::is_same_v<BDataType,
                               remove_cvref_t<typename BDramBlockWindow::DataType>>,
            "A/B block-window data types must match the pipeline problem");

        auto ab_lds_blocks = PipelineImplBase{}.GetABLdsTensorViews(p_smem);
        auto& a_lds_block  = ab_lds_blocks.get(number<0>{});
        auto& b_lds_block  = ab_lds_blocks.get(number<1>{});

        // Triton's global producer mapping is not CK's generic warp-raked
        // mapping. For both aligned RC operands, thread tid owns X vector
        // tid[2:0] and rows tid>>3 + {0,64,128,192}. Put the iteration factor
        // outside wave/lane in the logical Y dimension so producer registers,
        // the swizzled LDS addresses below, and MMAC consumers agree exactly.
        constexpr auto triton_full_dstr = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<4, 8, 8>, sequence<8, 8>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>>{});

        auto a_windows =
            PipelineImplBase{}.GetAWindows(a_dram_block_window, a_lds_block, triton_full_dstr);
        auto b_windows =
            PipelineImplBase{}.GetBWindows(b_dram_block_window, b_lds_block, triton_full_dstr);
        auto& a_dram_window = a_windows.get(number<0>{});
        auto& b_dram_window = b_windows.get(number<0>{});

        using ADramTileWindowStep = typename ADramBlockWindow::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindow::BottomTensorIndex;
        constexpr ADramTileWindowStep a_step =
            IsARow ? make_array(0, KPerBlock)
                   : make_array(KPerBlock, 0);
        constexpr BDramTileWindowStep b_step =
            IsBRow ? make_array(KPerBlock, 0)
                   : make_array(0, KPerBlock);

        [[maybe_unused]] constexpr auto b_stage_dstr =
            make_static_tile_distribution(StageBlockGemm::MakeBBlockDistributionEncode());
        [[maybe_unused]] constexpr auto a_stage_dstr =
            make_static_tile_distribution(StageBlockGemm::MakeABlockDistributionEncode());

        const auto store_packed_a = [&](const auto& a_tile) {
            // Exact Triton local_alloc swizzle for
            // vec=4/perPhase=1/maxPhase=16. The warp-raked global tiles hold
            // four half8 loads per operand; view them as eight half4 values
            // and pair the 8/16/24-KiB-separated destinations with st64.
            using AVec4 = ext_vector_t<ADataType, 4>;
            static_assert(remove_cvref_t<decltype(a_tile)>::get_thread_buffer_size() == 32,
                          "Triton input mapping requires 32 elements per "
                          "operand and thread");

            const auto& a_vecs =
                a_tile.get_thread_buffer().template get_as<AVec4>();
            const index_t tid = get_thread_id();
            const index_t store_base =
                ((tid << 4) & 8176) ^ (tid & 120);

            // CK's register buffer is [load0.low, load0.high, ...],
            // whereas Triton stores [all four lows, all four highs].
            StoreLdsPairSt64<0, 16>(store_base, a_vecs[0], a_vecs[2]);
            StoreLdsPairSt64<32, 48>(store_base, a_vecs[4], a_vecs[6]);
            StoreLdsPairSt64<0, 16>(store_base ^ 8, a_vecs[1], a_vecs[3]);
            StoreLdsPairSt64<32, 48>(store_base ^ 8, a_vecs[5], a_vecs[7]);
        };

        const auto store_packed_b = [&](const auto& b_tile) {
            using BVec4 = ext_vector_t<BDataType, 4>;
            static_assert(remove_cvref_t<decltype(b_tile)>::get_thread_buffer_size() == 32,
                          "Triton input mapping requires 32 elements per "
                          "operand and thread");
            const auto& b_vecs =
                b_tile.get_thread_buffer().template get_as<BVec4>();
            const index_t tid = get_thread_id();
            const index_t store_base =
                ((tid << 4) & 8176) ^ (tid & 120);
            // Fold B's 32-KiB half into the st64 immediate exactly as
            // Triton does. A and B now share the same two address VGPRs.
            StoreLdsPairSt64<64, 80>(
                store_base, b_vecs[0], b_vecs[2]);
            StoreLdsPairSt64<96, 112>(
                store_base, b_vecs[4], b_vecs[6]);
            StoreLdsPairSt64<64, 80>(
                store_base ^ 8, b_vecs[1], b_vecs[3]);
            StoreLdsPairSt64<96, 112>(
                store_base ^ 8, b_vecs[5], b_vecs[7]);
        };

        const auto store_a_transposed = [&](const auto& tile) {
            using Vec4 = ext_vector_t<ADataType, 4>;
            static_assert(remove_cvref_t<decltype(tile)>::get_thread_buffer_size() == 32,
                          "transposed producer requires eight K4 vectors");
            const auto& vecs = tile.get_thread_buffer().template get_as<Vec4>();
            const index_t tid = get_thread_id();
            const index_t m0  = (tid & 31) * 8;
            const index_t k   = (tid >> 5) * 4;
            static_for<0, 8, 1>{}([&](auto i) {
                const index_t m = m0 + i;
                const index_t concept_tid = (m % 64) * 8 + k / 8;
                const index_t base =
                    ((concept_tid << 4) & 8176) ^ (concept_tid & 120);
                const index_t byte_address =
                    (base ^ ((k % 8) / 4) * 8) + (m / 64) * 8192;
                StoreLdsB64(byte_address, vecs[i]);
            });
        };

        const auto store_b_transposed = [&](const auto& tile) {
            using Vec4 = ext_vector_t<BDataType, 4>;
            static_assert(remove_cvref_t<decltype(tile)>::get_thread_buffer_size() == 32,
                          "transposed producer requires eight K4 vectors");
            const auto& vecs = tile.get_thread_buffer().template get_as<Vec4>();
            const index_t tid = get_thread_id();
            const index_t n0  = (tid & 31) * 8;
            const index_t k   = (tid >> 5) * 4;
            static_for<0, 8, 1>{}([&](auto i) {
                const index_t n = n0 + i;
                const index_t concept_tid = (n % 64) * 8 + k / 8;
                const index_t base =
                    ((concept_tid << 4) & 8176) ^ (concept_tid & 120);
                const index_t byte_address =
                    32768 + (base ^ ((k % 8) / 4) * 8) +
                    (n / 64) * 8192;
                StoreLdsB64(byte_address, vecs[i]);
            });
        };

        const auto store_full_tile = [&](const auto& a_tile, const auto& b_tile) {
            if constexpr(IsARow)
            {
                store_packed_a(a_tile);
            }
            else
            {
                auto a_shuffled = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffled, a_tile);
                store_a_transposed(a_shuffled);
            }

            if constexpr(!IsBRow)
            {
                store_packed_b(b_tile);
            }
            else
            {
                auto b_shuffled = make_static_distributed_tensor<BDataType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffled, b_tile);
                store_b_transposed(b_shuffled);
            }
        };

        const auto stage_block_gemm = StageBlockGemm{};
        auto c_block_tile           = stage_block_gemm.MakeCBlockTile();
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        auto a_current_global = load_tile(a_dram_window);
        auto b_current_global = load_tile(b_dram_window);
        move_tile_window(a_dram_window, a_step);
        move_tile_window(b_dram_window, b_step);
        store_full_tile(a_current_global, b_current_global);
        block_sync_lds();

        [[maybe_unused]] const auto load_lds_stage =
            [&](auto iKStage,
                auto& a_stage_tile,
                auto& b_stage_tile) {
            static_assert(
                remove_cvref_t<decltype(a_stage_tile)>::get_thread_buffer_size() == 32 &&
                    remove_cvref_t<decltype(b_stage_tile)>::get_thread_buffer_size() == 16,
                "native M16xN16 stage requires eight A and four B half4 operands");

            // Exact Triton MMAC-consumer layout. A repeats are 4 KiB apart,
            // B repeats are 8 KiB apart, and successive K16 stages select
            // XOR-32 swizzle phases.
            const index_t tid = get_thread_id();
            const index_t a_load_base =
                ((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                 ((tid >> 1) & 24)) ^
                (iKStage * number<32>{});
            const index_t b_load_base =
                (((tid << 5) & 6144) |
                 (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                 ((tid & 15) << 7)) ^
                (iKStage * number<32>{});

            auto& a_pairs =
                a_stage_tile.get_thread_buffer().template get_as<fp32x4_t>();
            auto& b_pairs =
                b_stage_tile.get_thread_buffer().template get_as<fp32x4_t>();
            a_pairs[0] = LoadLdsPairSt64<0, 8>(a_load_base);
            a_pairs[1] = LoadLdsPairSt64<16, 24>(a_load_base);
            a_pairs[2] = LoadLdsPairSt64<32, 40>(a_load_base);
            a_pairs[3] = LoadLdsPairSt64<48, 56>(a_load_base);
            b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
            b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
        };

        [[maybe_unused]] const auto compute_lds_stage =
            [&](auto& a_stage_tile, auto& b_stage_tile) {
            stage_block_gemm(c_block_tile, a_stage_tile, b_stage_tile);
        };

        const auto compute_lds_tile =
            [&](auto&& late_global_load,
                [[maybe_unused]] auto is_final_tile) {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_THREE_SLOT_STREAMING) || \
    defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
            // Keep only one A pair and two B pairs live. This is the native
            // gfx936 operand lifetime for a 256x256/W8 tile: each A pair feeds
            // eight MMACs, then its four VGPRs are reused by the next M pair.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;
            using AVec = typename StageImpl::AVecType;
            using BVec = typename StageImpl::BVecType;
            static_assert(std::is_same_v<AVec, BVec>,
                          "three-slot streaming requires matching A/B vectors");

#if defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
            thread_buffer<fp32x4_t, 4> operand_slots;
#else
            thread_buffer<fp32x4_t, 3> operand_slots;
#endif

            const auto a_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                         ((tid >> 1) & 24)) ^
                        (iKStage * number<32>{}));
            };
            const auto b_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((tid << 5) & 6144) |
                        (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                        ((tid & 15) << 7)) ^
                       (iKStage * number<32>{});
            };
            const auto load_a_pair =
                [&](auto iKStage, auto iMPair, auto iASlot) {
                if constexpr(iMPair == number<0>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<0, 8>(a_base(iKStage));
                }
                else if constexpr(iMPair == number<1>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<16, 24>(a_base(iKStage));
                }
                else if constexpr(iMPair == number<2>{})
                {
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<32, 40>(a_base(iKStage));
                }
                else
                {
                    static_assert(iMPair == number<3>{});
                    operand_slots[iASlot] =
                        LoadLdsPairSt64<48, 56>(a_base(iKStage));
                }
            };
            const auto load_b_pairs = [&](auto iKStage) {
                operand_slots[number<1>{}] =
                    LoadLdsPairSt64<64, 80>(b_base(iKStage));
                operand_slots[number<2>{}] =
                    LoadLdsPairSt64<96, 112>(b_base(iKStage));
            };
            const auto compute_pair_head =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iNInPair) {
                    constexpr auto iMRepeat = iMPair * number<2>{};
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNInPair;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{}],
                                b_vecs[number<2>{} + iNInPair]);
                });
            };
            const auto compute_pair_tail =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iNInPair) {
                    constexpr auto iMRepeat = iMPair * number<2>{};
                    constexpr auto iNRepeat = number<2>{} + iNInPair;
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNRepeat;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{}],
                                b_vecs[number<4>{} + iNInPair]);
                });
                static_for<0, 4, 1>{}([&](auto iNRepeat) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + number<1>{};
                    constexpr auto c_index =
                        iMRepeat * number<4>{} + iNRepeat;
                    StageImpl{}(c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + number<1>{}],
                                b_vecs[number<2>{} + iNRepeat]);
                });
            };
            const auto compute_pair =
                [&](auto iMPair, auto iASlot) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs =
                    operand_slots.template get_as<AVec>();
                const auto& b_vecs =
                    operand_slots.template get_as<BVec>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto iMRepeat =
                            iMPair * number<2>{} + iMInPair;
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iASlot * number<2>{} + iMInPair],
                                    b_vecs[number<2>{} + iNRepeat]);
                    });
                });
            };

            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
#if defined(CK_TILE_GROUPED_GEMM_GFX936_FOUR_SLOT_STREAMING)
                // Double-buffer A pairs: issue the next ds_read before the
                // current eight MMACs so LDS latency is hidden without
                // retaining all four A pairs.
                load_a_pair(iKStage, number<0>{}, number<0>{});
                load_b_pairs(iKStage);
                load_a_pair(iKStage, number<1>{}, number<3>{});
                WaitLdsReads(number<2>{});
                compute_pair_head(number<0>{}, number<0>{});
                WaitLdsReads(number<1>{});
                compute_pair_tail(number<0>{}, number<0>{});

                WaitLdsReads();
                load_a_pair(iKStage, number<2>{}, number<0>{});
                compute_pair(number<1>{}, number<3>{});

                WaitLdsReads();
                load_a_pair(iKStage, number<3>{}, number<3>{});
                compute_pair(number<2>{}, number<0>{});

                WaitLdsReads();
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
                    if constexpr(is_final_tile.value)
                    {
                        block_sync_lds();
                        final_epilogue_pre_store(c_block_tile);
                    }
#endif
                }
                compute_pair(number<3>{}, number<3>{});
#else
                load_a_pair(
                    iKStage, number<0>{}, number<0>{});
                load_b_pairs(iKStage);
                // A0 and B0 are ready while the second B read may remain.
                WaitLdsReads(number<1>{});
                compute_pair_head(number<0>{}, number<0>{});
                WaitLdsReads();
                compute_pair_tail(number<0>{}, number<0>{});

                static_for<1, 4, 1>{}([&](auto iMPair) {
                    load_a_pair(
                        iKStage, iMPair, number<0>{});
                    WaitLdsReads();
                    if constexpr(iKStage == number<3>{} &&
                                 iMPair == number<3>{})
                    {
                        // Hide the next K64 global read behind the last eight
                        // MMACs, matching the native persistent hot loop.
                        late_global_load();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
                        if constexpr(is_final_tile.value)
                        {
                            // All input LDS reads are complete. Round 0 is
                            // already final, so use dead LDS while pair 3 runs.
                            block_sync_lds();
                            final_epilogue_pre_store(c_block_tile);
                        }
#endif
                    }
                    compute_pair(iMPair, number<0>{});
                });
#endif
            });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING)
            // Reproduce Triton's five-slot operand rotation. Each slot holds
            // one ds_read2st64_b64 result (two native MMAC operands). The
            // next K16 stage is read into a slot immediately after the
            // current operand dies, so only 20 operand VGPRs are live rather
            // than the 48 VGPRs required by two complete stage tensors.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;
            using AVec = typename StageImpl::AVecType;
            using BVec = typename StageImpl::BVecType;
            static_assert(std::is_same_v<AVec, BVec>,
                          "rolling operand slots require matching A/B vectors");

            thread_buffer<fp32x4_t, 5> operand_slots;

            const auto a_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                         ((tid >> 1) & 24)) ^
                        (iKStage * number<32>{}));
            };
            const auto b_base = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                return (((tid << 5) & 6144) |
                        (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                        ((tid & 15) << 7)) ^
                       (iKStage * number<32>{});
            };
            const auto load_a_slot =
                [&](auto iKStage, auto iMPair, auto iSlot) {
                    if constexpr(iMPair == number<0>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<0, 8>(a_base(iKStage));
                    }
                    else if constexpr(iMPair == number<1>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<16, 24>(a_base(iKStage));
                    }
                    else if constexpr(iMPair == number<2>{})
                    {
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<32, 40>(a_base(iKStage));
                    }
                    else
                    {
                        static_assert(iMPair == number<3>{});
                        operand_slots[iSlot] =
                            LoadLdsPairSt64<48, 56>(a_base(iKStage));
                    }
                };
            const auto load_b0_slot = [&](auto iKStage, auto iSlot) {
                operand_slots[iSlot] =
                    LoadLdsPairSt64<64, 80>(b_base(iKStage));
            };
            const auto load_b1_slot = [&](auto iKStage, auto iSlot) {
                operand_slots[iSlot] =
                    LoadLdsPairSt64<96, 112>(b_base(iKStage));
            };
            const auto compute_pair_half =
                [&](auto iMPair, auto iASlot, auto iBSlot, auto iNPair) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();
                    static_for<0, 2, 1>{}([&](auto iMInPair) {
                        static_for<0, 2, 1>{}([&](auto iNInPair) {
                            constexpr auto iMRepeat =
                                iMPair * number<2>{} + iMInPair;
                            constexpr auto iNRepeat =
                                iNPair * number<2>{} + iNInPair;
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(
                                c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + iMInPair],
                                b_vecs[iBSlot * number<2>{} + iNInPair]);
                        });
                    });
                };
            const auto compute_pair =
                [&](auto iMPair,
                    auto iASlot,
                    auto iB0Slot,
                    auto iB1Slot) {
                    compute_pair_half(
                        iMPair, iASlot, iB0Slot, number<0>{});
                    compute_pair_half(
                        iMPair, iASlot, iB1Slot, number<1>{});
                };
            const auto compute_pair_head =
                [&](auto iMPair, auto iASlot, auto iB0Slot) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();
                    static_for<0, 2, 1>{}([&](auto iNInPair) {
                        constexpr auto iMRepeat = iMPair * number<2>{};
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNInPair;
                        StageImpl{}(
                            c_vecs[c_index],
                            a_vecs[iASlot * number<2>{}],
                            b_vecs[iB0Slot * number<2>{} + iNInPair]);
                    });
                };
            const auto compute_pair_tail =
                [&](auto iMPair,
                    auto iASlot,
                    auto iB0Slot,
                    auto iB1Slot) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        operand_slots.template get_as<AVec>();
                    const auto& b_vecs =
                        operand_slots.template get_as<BVec>();

                    // Triton's gfx936 hot loop starts after A0 and B0 are
                    // ready, then waits for B1 before consuming the remaining
                    // six native operands. Preserve that 2+6 issue order
                    // instead of CK's otherwise equivalent 4+4 grouping.
                    static_for<0, 2, 1>{}([&](auto iNInPair) {
                        constexpr auto iMRepeat = iMPair * number<2>{};
                        constexpr auto iNRepeat =
                            number<2>{} + iNInPair;
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(
                            c_vecs[c_index],
                            a_vecs[iASlot * number<2>{}],
                            b_vecs[iB1Slot * number<2>{} + iNInPair]);
                    });
                    static_for<0, 2, 1>{}([&](auto iNPair) {
                        static_for<0, 2, 1>{}([&](auto iNInPair) {
                            constexpr auto iMRepeat =
                                iMPair * number<2>{} + number<1>{};
                            constexpr auto iNRepeat =
                                iNPair * number<2>{} + iNInPair;
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            constexpr auto iBSlot =
                                iNPair == number<0>{} ? iB0Slot : iB1Slot;
                            StageImpl{}(
                                c_vecs[c_index],
                                a_vecs[iASlot * number<2>{} + number<1>{}],
                                b_vecs[iBSlot * number<2>{} + iNInPair]);
                        });
                    });
                };

            // Stage 0: S0=A0, S1=B0, S2=B1, S3=A1, S4=A2.
            load_a_slot(number<0>{}, number<0>{}, number<0>{});
            load_b0_slot(number<0>{}, number<1>{});
            load_b1_slot(number<0>{}, number<2>{});
            load_a_slot(number<0>{}, number<1>{}, number<3>{});
            load_a_slot(number<0>{}, number<2>{}, number<4>{});
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<0>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<0>{}, number<1>{}, number<2>{});
#else
            compute_pair_half(
                number<0>{}, number<0>{}, number<1>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<0>{}, number<2>{}, number<1>{});
#endif
            load_a_slot(number<0>{}, number<3>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<3>{}, number<1>{}, number<2>{});
            load_a_slot(number<1>{}, number<0>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<4>{}, number<1>{}, number<2>{});
            load_b0_slot(number<1>{}, number<4>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<1>{}, number<0>{});
            load_b1_slot(number<1>{}, number<1>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<2>{}, number<1>{});
            load_a_slot(number<1>{}, number<1>{}, number<0>{});
            load_a_slot(number<1>{}, number<2>{}, number<2>{});

            // Stage 1: S3=A0, S4=B0, S1=B1, S0=A1, S2=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<3>{}, number<4>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<3>{}, number<4>{}, number<1>{});
#else
            compute_pair_half(
                number<0>{}, number<3>{}, number<4>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<3>{}, number<1>{}, number<1>{});
#endif
            load_a_slot(number<1>{}, number<3>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<0>{}, number<4>{}, number<1>{});
            load_a_slot(number<2>{}, number<0>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<2>{}, number<4>{}, number<1>{});
            load_b0_slot(number<2>{}, number<2>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<3>{}, number<4>{}, number<0>{});
            load_b1_slot(number<2>{}, number<4>{});
            compute_pair_half(
                number<3>{}, number<3>{}, number<1>{}, number<1>{});
            load_a_slot(number<2>{}, number<1>{}, number<3>{});
            load_a_slot(number<2>{}, number<2>{}, number<1>{});

            // Stage 2: S0=A0, S2=B0, S4=B1, S3=A1, S1=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<0>{}, number<2>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<0>{}, number<2>{}, number<4>{});
#else
            compute_pair_half(
                number<0>{}, number<0>{}, number<2>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<0>{}, number<4>{}, number<1>{});
#endif
            load_a_slot(number<2>{}, number<3>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<3>{}, number<2>{}, number<4>{});
            load_a_slot(number<3>{}, number<0>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<2>{}, number<1>{}, number<2>{}, number<4>{});
            load_b0_slot(number<3>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<2>{}, number<0>{});
            load_b1_slot(number<3>{}, number<2>{});
            compute_pair_half(
                number<3>{}, number<0>{}, number<4>{}, number<1>{});
            load_a_slot(number<3>{}, number<1>{}, number<0>{});
            load_a_slot(number<3>{}, number<2>{}, number<4>{});

            // Stage 3: S3=A0, S1=B0, S2=B1, S0=A1, S4=A2.
            WaitLdsReads(number<3>{});
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_ROLLING_EXACT_WAIT)
            compute_pair_head(
                number<0>{}, number<3>{}, number<1>{});
            WaitLdsReads(number<2>{});
            compute_pair_tail(
                number<0>{}, number<3>{}, number<1>{}, number<2>{});
#else
            compute_pair_half(
                number<0>{}, number<3>{}, number<1>{}, number<0>{});
            WaitLdsReads(number<2>{});
            compute_pair_half(
                number<0>{}, number<3>{}, number<2>{}, number<1>{});
#endif
            load_a_slot(number<3>{}, number<3>{}, number<3>{});
            WaitLdsReads(number<2>{});
            compute_pair(
                number<1>{}, number<0>{}, number<1>{}, number<2>{});
            WaitLdsReads(number<1>{});
            compute_pair(
                number<2>{}, number<4>{}, number<1>{}, number<2>{});
            // On the split-global variant, match Triton's aligned hot loop:
            // issue the next B K64 tile only after the third A pair dies and
            // hide those four VMEM reads behind the final eight MMACs.
            late_global_load();
            WaitLdsReads();
            compute_pair(
                number<3>{}, number<3>{}, number<1>{}, number<2>{});
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SINGLE_PARTIAL)
            // One operand set, consumed in issue order. This keeps the
            // 221-VGPR baseline footprint while replacing each all-six-read
            // wait with Triton-style partial waits.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_stage_interleaved = [&](auto iKStage) {
                const index_t tid = get_thread_id();
                const index_t a_load_base =
                    (((((tid & 15) << 7) |
                       ((tid << 3) & 2168)) ^
                      ((tid >> 1) & 24)) ^
                     (iKStage * number<32>{}));
                const index_t b_load_base =
                    (((tid << 5) & 6144) |
                     (((tid & 15) << 3) ^
                      ((tid >> 1) & 24)) |
                     ((tid & 15) << 7)) ^
                    (iKStage * number<32>{});
                auto& a_pairs = a_stage.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                auto& b_pairs = b_stage.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                a_pairs[0] = LoadLdsPairSt64<0, 8>(a_load_base);
                b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
                a_pairs[1] = LoadLdsPairSt64<16, 24>(a_load_base);
                a_pairs[2] = LoadLdsPairSt64<32, 40>(a_load_base);
                a_pairs[3] = LoadLdsPairSt64<48, 56>(a_load_base);
            };

            const auto compute_m_pair = [&](auto iMPair) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs = a_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::AVecType>();
                const auto& b_vecs = b_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::BVecType>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + iMInPair;
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iMRepeat],
                                    b_vecs[iNRepeat]);
                    });
                });
            };

            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                load_stage_interleaved(iKStage);
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
                }
                WaitLdsReads(number<3>{});
                compute_m_pair(number<0>{});
                WaitLdsReads(number<2>{});
                compute_m_pair(number<1>{});
                WaitLdsReads(number<1>{});
                compute_m_pair(number<2>{});
                WaitLdsReads();
                compute_m_pair(number<3>{});
            });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_OVERLAP)
            // Keep two operand sets, but begin stage-0 MMACs once A pair 0
            // and both B pairs are ready. Stage 1 is already in flight, so
            // descending waits reproduce Triton's lgkmcnt-driven consumption
            // instead of stalling for all twelve read2 instructions.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage0 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto a_stage1 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_lds_stage_interleaved =
                [&](auto iKStage,
                    auto& a_stage_tile,
                    auto& b_stage_tile) {
                    const index_t tid = get_thread_id();
                    const index_t a_load_base =
                        (((((tid & 15) << 7) |
                           ((tid << 3) & 2168)) ^
                          ((tid >> 1) & 24)) ^
                         (iKStage * number<32>{}));
                    const index_t b_load_base =
                        (((tid << 5) & 6144) |
                         (((tid & 15) << 3) ^
                          ((tid >> 1) & 24)) |
                         ((tid & 15) << 7)) ^
                        (iKStage * number<32>{});
                    auto& a_pairs = a_stage_tile.get_thread_buffer()
                                        .template get_as<fp32x4_t>();
                    auto& b_pairs = b_stage_tile.get_thread_buffer()
                                        .template get_as<fp32x4_t>();
                    a_pairs[0] =
                        LoadLdsPairSt64<0, 8>(a_load_base);
                    b_pairs[0] =
                        LoadLdsPairSt64<64, 80>(b_load_base);
                    b_pairs[1] =
                        LoadLdsPairSt64<96, 112>(b_load_base);
                    a_pairs[1] =
                        LoadLdsPairSt64<16, 24>(a_load_base);
                    a_pairs[2] =
                        LoadLdsPairSt64<32, 40>(a_load_base);
                    a_pairs[3] =
                        LoadLdsPairSt64<48, 56>(a_load_base);
                };

            const auto compute_m_pair =
                [&](auto iMPair,
                    auto& a_stage_tile,
                    auto& b_stage_tile) {
                    auto& c_vecs = c_block_tile.get_thread_buffer()
                                       .template get_as<
                                           typename StageImpl::CVecType>();
                    const auto& a_vecs =
                        a_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::AVecType>();
                    const auto& b_vecs =
                        b_stage_tile.get_thread_buffer()
                            .template get_as<typename StageImpl::BVecType>();
                    static_for<0, 2, 1>{}([&](auto iMInPair) {
                        constexpr auto iMRepeat =
                            iMPair * number<2>{} + iMInPair;
                        static_for<0, 4, 1>{}([&](auto iNRepeat) {
                            constexpr auto c_index =
                                iMRepeat * number<4>{} + iNRepeat;
                            StageImpl{}(c_vecs[c_index],
                                        a_vecs[iMRepeat],
                                        b_vecs[iNRepeat]);
                        });
                    });
                };

            load_lds_stage_interleaved(
                number<0>{}, a_stage0, b_stage0);
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_PARTIAL_LATE)
            WaitLdsReads(number<3>{});
            compute_m_pair(number<0>{}, a_stage0, b_stage0);
            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            WaitLdsReads(number<8>{});
            compute_m_pair(number<1>{}, a_stage0, b_stage0);
            WaitLdsReads(number<7>{});
            compute_m_pair(number<2>{}, a_stage0, b_stage0);
            WaitLdsReads(number<6>{});
            compute_m_pair(number<3>{}, a_stage0, b_stage0);
#else
            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            WaitLdsReads(number<9>{});
            compute_m_pair(number<0>{}, a_stage0, b_stage0);
            WaitLdsReads(number<8>{});
            compute_m_pair(number<1>{}, a_stage0, b_stage0);
            WaitLdsReads(number<7>{});
            compute_m_pair(number<2>{}, a_stage0, b_stage0);
            WaitLdsReads(number<6>{});
            compute_m_pair(number<3>{}, a_stage0, b_stage0);
#endif
            WaitLdsReads();

            load_lds_stage(number<2>{}, a_stage0, b_stage0);
            compute_lds_stage(a_stage1, b_stage1);
            WaitLdsReads();
            load_lds_stage(number<3>{}, a_stage1, b_stage1);
            compute_lds_stage(a_stage0, b_stage0);
            late_global_load();
            WaitLdsReads();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_EPILOGUE_PRE_SYNC)
            // All final-K16 operands now reside in VGPRs. Move the first
            // epilogue barrier here so no wave can overwrite input LDS while
            // another still reads it, while leaving final MMAC and output
            // conversion in one scheduler region.
            block_sync_lds();
#endif
#if defined(CK_TILE_GROUPED_GEMM_TRITON_FUSED_EPILOGUE)
            if constexpr(is_final_tile.value)
            {
                // All final K16 operands are resident in VGPRs. Once every
                // wave has stopped reading A/B LDS, finish the first M pair,
                // convert its 32 accumulator scalars, and write epilogue
                // round 0 into the dead compute allocation. The remaining
                // three M pairs cover the four LDS writes.
                block_sync_lds();
                compute_m_pair(number<0>{}, a_stage1, b_stage1);
                final_epilogue_pre_store(c_block_tile);
                compute_m_pair(number<1>{}, a_stage1, b_stage1);
                compute_m_pair(number<2>{}, a_stage1, b_stage1);
                compute_m_pair(number<3>{}, a_stage1, b_stage1);
            }
            else
            {
                compute_lds_stage(a_stage1, b_stage1);
            }
#else
            compute_lds_stage(a_stage1, b_stage1);
#endif
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_INPLACE_OVERLAP)
            // Triton-like operand lifetime control. A pair is dead after its
            // eight MMACs (two M repeats x four N repeats), so immediately
            // overwrite those four VGPRs with the same pair from the next
            // K16 stage. Only B needs a second register set.
            using StageWarpGemm = typename StageBlockGemm::WarpGemm;
            using StageImpl =
                typename StageWarpGemm::WarpGemmAttribute::Impl;

            auto a_stage =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            const auto load_b_stage = [&](auto iKStage,
                                          auto& b_stage_tile) {
                const index_t tid = get_thread_id();
                const index_t b_load_base =
                    (((tid << 5) & 6144) |
                     (((tid & 15) << 3) ^ ((tid >> 1) & 24)) |
                     ((tid & 15) << 7)) ^
                    (iKStage * number<32>{});
                auto& b_pairs = b_stage_tile.get_thread_buffer()
                                    .template get_as<fp32x4_t>();
                b_pairs[0] = LoadLdsPairSt64<64, 80>(b_load_base);
                b_pairs[1] = LoadLdsPairSt64<96, 112>(b_load_base);
            };

            const auto compute_a_pair = [&](auto iMPair,
                                             auto& b_stage_tile) {
                auto& c_vecs = c_block_tile.get_thread_buffer()
                                   .template get_as<typename StageImpl::CVecType>();
                const auto& a_vecs = a_stage.get_thread_buffer()
                                         .template get_as<typename StageImpl::AVecType>();
                const auto& b_vecs = b_stage_tile.get_thread_buffer()
                                         .template get_as<typename StageImpl::BVecType>();
                static_for<0, 2, 1>{}([&](auto iMInPair) {
                    constexpr auto iMRepeat =
                        iMPair * number<2>{} + iMInPair;
                    static_for<0, 4, 1>{}([&](auto iNRepeat) {
                        constexpr auto c_index =
                            iMRepeat * number<4>{} + iNRepeat;
                        StageImpl{}(c_vecs[c_index],
                                    a_vecs[iMRepeat],
                                    b_vecs[iNRepeat]);
                    });
                });
            };

            const auto reload_a_pair = [&](auto iKStage, auto iMPair) {
                const index_t tid = get_thread_id();
                const index_t a_load_base =
                    (((((tid & 15) << 7) | ((tid << 3) & 2168)) ^
                      ((tid >> 1) & 24)) ^
                     (iKStage * number<32>{})) +
                    iMPair * number<8192>{};
                auto& a_pairs =
                    a_stage.get_thread_buffer().template get_as<fp32x4_t>();
                a_pairs[iMPair] =
                    LoadLdsPairSt64<0, 8>(a_load_base);
            };

            const auto compute_and_prefetch_next =
                [&](auto iNextKStage,
                    auto& b_current,
                    auto& b_next) {
                    // B remains live for all 32 current-stage MMACs, so issue
                    // the two next-stage B reads into a small second set.
                    load_b_stage(iNextKStage, b_next);
                    static_for<0, 4, 1>{}([&](auto iMPair) {
                        compute_a_pair(iMPair, b_current);
                        reload_a_pair(iNextKStage, iMPair);
                    });
                    WaitLdsReads();
                };

            load_lds_stage(number<0>{}, a_stage, b_stage0);
            WaitLdsReads();
            compute_and_prefetch_next(
                number<1>{}, b_stage0, b_stage1);
            compute_and_prefetch_next(
                number<2>{}, b_stage1, b_stage0);
            compute_and_prefetch_next(
                number<3>{}, b_stage0, b_stage1);
            late_global_load();
            static_for<0, 4, 1>{}(
                [&](auto iMPair) { compute_a_pair(iMPair, b_stage1); });
#elif defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_OVERLAP)
            // Ping-pong two operand register sets. Once stage S is ready,
            // issue LDS reads for S+1 and cover their latency with the 32
            // native MMACs of S. Reuse a register set only after its MMACs
            // have consumed it, bounding the live range below the spill
            // threshold.
            auto a_stage0 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage0 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);
            auto a_stage1 =
                make_static_distributed_tensor<ADataType>(a_stage_dstr);
            auto b_stage1 =
                make_static_distributed_tensor<BDataType>(b_stage_dstr);

            load_lds_stage(number<0>{}, a_stage0, b_stage0);
            WaitLdsReads();

            load_lds_stage(number<1>{}, a_stage1, b_stage1);
            compute_lds_stage(a_stage0, b_stage0);
            WaitLdsReads();

            load_lds_stage(number<2>{}, a_stage0, b_stage0);
            compute_lds_stage(a_stage1, b_stage1);
            WaitLdsReads();

            load_lds_stage(number<3>{}, a_stage1, b_stage1);
            compute_lds_stage(a_stage0, b_stage0);
            late_global_load();
            WaitLdsReads();
#if defined(CK_TILE_GROUPED_GEMM_TRITON_EPILOGUE_PRE_SYNC)
            // All final-K16 operands now reside in VGPRs. Move the first
            // epilogue barrier here so no wave can overwrite input LDS while
            // another still reads it, while leaving final MMAC and output
            // conversion in one scheduler region.
            block_sync_lds();
#endif
            compute_lds_stage(a_stage1, b_stage1);
#else
            static_for<0, NumKStages, 1>{}([&](auto iKStage) {
                auto a_stage_tile =
                    make_static_distributed_tensor<ADataType>(a_stage_dstr);
                auto b_stage_tile =
                    make_static_distributed_tensor<BDataType>(b_stage_dstr);
                load_lds_stage(iKStage, a_stage_tile, b_stage_tile);
                WaitLdsReads();
                if constexpr(iKStage == number<3>{})
                {
                    late_global_load();
                }
                compute_lds_stage(a_stage_tile, b_stage_tile);
            });
#endif
        };

        index_t i = 1;
        while(i < num_loop)
        {
#if defined(CK_TILE_GROUPED_GEMM_TRITON_OPERAND_SPLIT_GLOBAL)
            // Triton issues A's next-tile VMEM loads before the K16 operand
            // loop, but delays B until the final K16 MMAC stage. This lets
            // the dead first operand set back B's registers and overlaps B
            // VMEM latency with the final 32 MMACs.
            auto a_next_global = load_tile(a_dram_window);
            move_tile_window(a_dram_window, a_step);
            decltype(load_tile(b_dram_window)) b_next_global;

            compute_lds_tile(
                [&]() {
                    b_next_global = load_tile(b_dram_window);
                    move_tile_window(b_dram_window, b_step);
                },
                bool_constant<false>{});
#else
            // Issue the next global K64 tile before consuming the current LDS
            // tile. Its registers stay live across four K16 MMAC steps.
            auto a_next_global = load_tile(a_dram_window);
            auto b_next_global = load_tile(b_dram_window);
            move_tile_window(a_dram_window, a_step);
            move_tile_window(b_dram_window, b_step);

            compute_lds_tile([&]() {}, bool_constant<false>{});
#endif
            block_sync_lds();
            store_full_tile(a_next_global, b_next_global);
            block_sync_lds();
            ++i;
        }

        compute_lds_tile([&]() {}, bool_constant<true>{});
        // The candidate epilogue consumes the native MMAC accumulator layout
        // and performs the lane transpose through the now-dead A/B LDS. Keep
        // the raw tile here so the producer stores can use the packed MMAC
        // register order instead of first scalarizing CWarpOutputDstr.
        return c_block_tile;
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const auto no_final_epilogue_pre_store = [](const auto&) {};
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   no_final_epilogue_pre_store);
    }

    template <typename ADramBlockWindow, typename BDramBlockWindow>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindow& a_dram_block_window,
                                   const BDramBlockWindow& b_dram_block_window,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem) const
    {
        const auto no_final_epilogue_pre_store = [](const auto&) {};
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   no_final_epilogue_pre_store);
    }

    template <typename ADramBlockWindow,
              typename BDramBlockWindow,
              typename FinalEpiloguePreStore>
    CK_TILE_DEVICE auto RunWithEpiloguePreStore(
        const ADramBlockWindow& a_dram_block_window,
        const BDramBlockWindow& b_dram_block_window,
        index_t num_loop,
        void* p_smem,
        FinalEpiloguePreStore&& final_epilogue_pre_store) const
    {
        return Run(a_dram_block_window,
                   b_dram_block_window,
                   num_loop,
                   p_smem,
                   static_cast<FinalEpiloguePreStore&&>(
                       final_epilogue_pre_store));
    }
};

} // namespace ck_tile
