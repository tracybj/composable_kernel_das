// SPDX-License-Identifier: MIT
#pragma once

#include "grouped_int8_nt_problem.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "grouped_int8_nt_wasp_pipeline.hpp"

namespace ck_tile {

// -----------------------------------------------------------------------------
// 1. Core Parameterized WASP Problem (supports BM=32/64/128/256, BN=32*ConsumerWaves)
// -----------------------------------------------------------------------------
template <int Rows_,
          bool Transpose_    = false,
          int Split_         = 8,
          int Group_         = 4,
          int Batch_         = 4,
          bool Prefetch_     = true,
          bool OneBarrier_   = true,
          int RecordCap_     = 0,
          bool EarlyB_       = false,
          bool PackedB_      = false,
          int BAddress_      = 0,
          int ConsumerWaves_ = 8,
          bool MaskInactiveN_ = false>
struct GroupedInt8NtWaspCoreProblem : GroupedInt8NtTileProblem<Rows_,
                                                               32 * ConsumerWaves_,
                                                               128,
                                                               1,
                                                               ConsumerWaves_,
                                                               true,
                                                               true,
                                                               true,
                                                               false,
                                                               1,
                                                               true,
                                                               Group_>
{
    static constexpr bool Transpose = Transpose_, PackedB = PackedB_;
    static constexpr index_t ProducerWaves  = (Rows_ <= 64 ? 1 : Rows_ / 64),
                             ComputeThreads = 64 * ConsumerWaves_,
                             Threads        = 64 * (ProducerWaves + ConsumerWaves_);
    using Shape           = TileGemmShape<sequence<Rows_, 32 * ConsumerWaves_, 128>,
                                          sequence<1, ConsumerWaves_, 1>,
                                          sequence<16, 16, 32>>;
    using Row             = tensor_layout::gemm::RowMajor;
    using Col             = tensor_layout::gemm::ColumnMajor;
    using Traits          = PersistentTileGemmUniversalTraits<true,
                                                              false,
                                                              false,
                                                              false,
                                                              Row,
                                                              Col,
                                                              Row,
                                                              false,
                                                              false,
                                                              false,
                                                              true,
                                                              false,
                                                              false>;
    using PipelineProblem = UniversalGemmPipelineProblem<int8_t,
                                                         int8_t,
                                                         int32_t,
                                                         Shape,
                                                         Traits,
                                                         GemmPipelineScheduler::Intrawave,
                                                         true,
                                                         TailNumber::Full,
                                                         null_type,
                                                         null_type,
                                                         false,
                                                         2,
                                                         2>;
    using Pipeline        = GroupedInt8NtWaspPipeline<PipelineProblem,
                                                      GemmPipelineAgBgCrCompV4DefaultPolicy,
                                                      Split_,
                                                      Batch_,
                                                      Prefetch_,
                                                      OneBarrier_,
                                                      RecordCap_,
                                                      EarlyB_,
                                                      PackedB_,
                                                      BAddress_,
                                                      MaskInactiveN_>;
    static_assert(Pipeline::BlockSize == Threads && Pipeline::kWaspLdsBytes == 2 * Rows_ * 128);
};

// -----------------------------------------------------------------------------
// 2. Standard & Subtile WASP Problem Aliases
// -----------------------------------------------------------------------------

// Legacy / Standard M256 Raw WASP Problem
template <bool Transpose_  = false,
          int Split_       = 8,
          int Group_       = 4,
          int Batch_       = 4,
          bool Prefetch_   = true,
          bool OneBarrier_ = true,
          int RecordCap_   = 0>
using GroupedInt8NtWaspProblem = GroupedInt8NtWaspCoreProblem<256,
                                                              Transpose_,
                                                              Split_,
                                                              Group_,
                                                              Batch_,
                                                              Prefetch_,
                                                              OneBarrier_,
                                                              RecordCap_>;

// Standard M256 Tile WASP Problem (Cached & Raw)
template <bool Transpose_  = false,
          int Split_       = 8,
          int Group_       = 4,
          int Batch_       = 4,
          bool Prefetch_   = true,
          bool OneBarrier_ = true,
          int RecordCap_   = 0,
          bool EarlyB_     = false,
          bool PackedB_    = false,
          int BAddress_    = 0>
using GroupedInt8NtWaspCachedProblem = GroupedInt8NtWaspCoreProblem<256,
                                                                    Transpose_,
                                                                    Split_,
                                                                    Group_,
                                                                    Batch_,
                                                                    Prefetch_,
                                                                    OneBarrier_,
                                                                    RecordCap_,
                                                                    EarlyB_,
                                                                    PackedB_,
                                                                    BAddress_,
                                                                    8>;

// M128 Split-M Subtile WASP Problem
template <bool Transpose_    = false,
          int Split_         = 8,
          int Group_         = 4,
          int Batch_         = 4,
          bool Prefetch_     = true,
          bool OneBarrier_   = true,
          int RecordCap_     = 0,
          bool EarlyB_       = false,
          bool PackedB_      = false,
          int BAddress_      = 0,
          int ConsumerWaves_ = 8>
using GroupedInt8NtWaspSplitMProblem = GroupedInt8NtWaspCoreProblem<128,
                                                                    Transpose_,
                                                                    Split_,
                                                                    Group_,
                                                                    Batch_,
                                                                    Prefetch_,
                                                                    OneBarrier_,
                                                                    RecordCap_,
                                                                    EarlyB_,
                                                                    PackedB_,
                                                                    BAddress_,
                                                                    ConsumerWaves_>;

// M64 Small-M Subtile WASP Problem
template <bool Transpose_    = false,
          int Split_         = 8,
          int Group_         = 4,
          int Batch_         = 4,
          bool Prefetch_     = true,
          bool OneBarrier_   = true,
          int RecordCap_     = 0,
          bool EarlyB_       = false,
          bool PackedB_      = false,
          int BAddress_      = 0,
          int ConsumerWaves_ = 8>
using GroupedInt8NtWaspSmallMProblem = GroupedInt8NtWaspCoreProblem<64,
                                                                    Transpose_,
                                                                    Split_,
                                                                    Group_,
                                                                    Batch_,
                                                                    Prefetch_,
                                                                    OneBarrier_,
                                                                    RecordCap_,
                                                                    EarlyB_,
                                                                    PackedB_,
                                                                    BAddress_,
                                                                    ConsumerWaves_>;

// M32 Small-M Subtile WASP Problem
template <bool Transpose_    = false,
          int Split_         = 8,
          int Group_         = 4,
          int Batch_         = 2,
          bool Prefetch_     = true,
          bool OneBarrier_   = true,
          int RecordCap_     = 0,
          bool EarlyB_       = false,
          bool PackedB_      = false,
          int BAddress_      = 0,
          int ConsumerWaves_ = 8>
using GroupedInt8NtWaspM32Problem = GroupedInt8NtWaspCoreProblem<32,
                                                                 Transpose_,
                                                                 Split_,
                                                                 Group_,
                                                                 Batch_,
                                                                 Prefetch_,
                                                                 OneBarrier_,
                                                                 RecordCap_,
                                                                 EarlyB_,
                                                                 PackedB_,
                                                                 BAddress_,
                                                                 ConsumerWaves_>;

// Wide-N Tile WASP Problem
template <bool Transpose_    = false,
          int Split_         = 8,
          int Group_         = 4,
          int Batch_         = 4,
          bool Prefetch_     = true,
          bool OneBarrier_   = true,
          int RecordCap_     = 0,
          bool EarlyB_       = false,
          bool PackedB_      = false,
          int BAddress_      = 0,
          int ConsumerWaves_ = 8>
using GroupedInt8NtWaspPublicWideNProblem = GroupedInt8NtWaspCoreProblem<256,
                                                                          Transpose_,
                                                                          Split_,
                                                                          Group_,
                                                                          Batch_,
                                                                          Prefetch_,
                                                                          OneBarrier_,
                                                                          RecordCap_,
                                                                          EarlyB_,
                                                                          PackedB_,
                                                                          BAddress_,
                                                                          ConsumerWaves_>;

// Prepared B, original NT direction. Entirely absent N stripes retain all
// main-loop barrier epochs and are discarded before any epilogue accumulator
// read.
template <int MGroup_>
using GroupedInt8NtWaspMaskedWideProblem =
    GroupedInt8NtWaspCoreProblem<256, false, 12, MGroup_, 2, true, true, 0, false, true, 0, 12, true>;

// The same masked geometry with high-half A issued during the final low-half
// MMAC batch. The original type remains unchanged for the INT32 entry point.
template <int MGroup_>
struct GroupedInt8NtWaspScaledMaskedPrefetchProblem : GroupedInt8NtWaspMaskedWideProblem<MGroup_>
{
    using Base     = GroupedInt8NtWaspMaskedWideProblem<MGroup_>;
    using Pipeline = GroupedInt8NtWaspPipeline<typename Base::PipelineProblem,
                                               GemmPipelineAgBgCrCompV4DefaultPolicy,
                                               12,
                                               2,
                                               true,
                                               true,
                                               0,
                                               false,
                                               true,
                                               0,
                                               true,
                                               false,
                                               true>;
    static_assert(Pipeline::BlockSize == Base::Threads && Pipeline::kWaspLdsBytes == 65536);
};

} // namespace ck_tile
