// SPDX-License-Identifier: MIT
#pragma once
#include "../core_problem.hpp"

namespace ck_tile {
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
