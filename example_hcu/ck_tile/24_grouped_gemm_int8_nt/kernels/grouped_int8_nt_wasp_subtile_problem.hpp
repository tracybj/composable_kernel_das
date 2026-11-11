// SPDX-License-Identifier: MIT
#pragma once

#include "grouped_int8_nt_wasp_core_problem.hpp"

namespace ck_tile {

// Standard M256 Tile WASP Problem
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
// read. This is an instance policy of the existing WASP core.
template <int MGroup_>
struct GroupedInt8NtWaspMaskedWideProblem
    : GroupedInt8NtWaspCoreProblem<256, false, 12, MGroup_, 2, true, true, 0, false, true, 0, 12>
{
    using Base =
        GroupedInt8NtWaspCoreProblem<256, false, 12, MGroup_, 2, true, true, 0, false, true, 0, 12>;
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
                                               true>;
    static_assert(Pipeline::BlockSize == Base::Threads && Pipeline::kWaspLdsBytes == 65536);
};

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
