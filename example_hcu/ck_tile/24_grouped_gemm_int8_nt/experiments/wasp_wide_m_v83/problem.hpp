// SPDX-License-Identifier: MIT
#pragma once
#include "../../grouped_int8_nt_wasp_problem.hpp"
#include "pipeline.hpp"
namespace ck_tile {
template<bool Transpose_ = false, int Batch_ = 4, int RecordCap_ = 0, int Group_ = 4>
struct GroupedInt8NtWaspWideProblem
    : GroupedInt8NtTileProblem<512, 256, 64, 1, 8, true, true, true, false, 1, true, Group_>
{
    static constexpr bool Transpose = Transpose_;
    static constexpr index_t ProducerWaves = 4, ComputeThreads = 512, Threads = 768;
    using Shape = TileGemmShape<sequence<512, 256, 64>, sequence<1, 8, 1>, sequence<16, 16, 32>>;
    using Row = tensor_layout::gemm::RowMajor;
    using Col = tensor_layout::gemm::ColumnMajor;
    using Traits = PersistentTileGemmUniversalTraits<true, false, false, false, Row, Col, Row,
                                                    false, false, false, true, false, false>;
    using PipelineProblem = UniversalGemmPipelineProblem<int8_t, int8_t, int32_t, Shape, Traits,
        GemmPipelineScheduler::Intrawave, true, TailNumber::Full, null_type, null_type, false, 2, 2>;
    using Pipeline = GemmPipelineInt8WaspWide<PipelineProblem, Batch_, RecordCap_>;
    static_assert(Pipeline::BlockSize == Threads && Pipeline::kWaspLdsBytes == 65536);
};
} // namespace ck_tile
