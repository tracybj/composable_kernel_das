// SPDX-License-Identifier: MIT
#pragma once
#include "pipeline/wasp.hpp"
namespace ck_tile {
using Int8WaspAccTile = Int8WaspAccTileCore<256>;
namespace grouped_int8_nt_wasp_detail = grouped_int8_nt_wasp_core_detail;
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy,
          index_t ConsumerSplit = 8, index_t AReadBatch = 4, bool APrefetch = true,
          bool OneBarrier = true, index_t RecordCap = 0>
using GemmPipelineInt8Wasp = GroupedInt8NtWaspPipeline<Problem, Policy,
    ConsumerSplit, AReadBatch, APrefetch, OneBarrier, RecordCap>;
}
