// SPDX-License-Identifier: MIT
#pragma once
#include "../pipeline/wasp.hpp"
namespace ck_tile {
using Int8WaspCachedAccTile = Int8WaspAccTileCore<256>;
namespace grouped_int8_nt_wasp_cached_detail = grouped_int8_nt_wasp_core_detail;
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy, index_t ConsumerSplit = 8, index_t AReadBatch = 1, bool APrefetch = false, bool OneBarrier = false, index_t RecordCap = 0, bool EarlyB = false, bool PackedB = false, index_t BAddress = 0>
using GemmPipelineInt8WaspCached = GroupedInt8NtWaspPipeline<Problem, Policy,
    ConsumerSplit, AReadBatch, APrefetch, OneBarrier, RecordCap, EarlyB, PackedB, BAddress>;
}
