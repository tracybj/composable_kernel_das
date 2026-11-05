"""One small-M cache-hint control, not a geometry/cache Cartesian sweep."""
from pathlib import Path
import json
R=Path('/workspace/composable_kernel');I=R/'include/ck_tile/ops/gemm/pipeline';G=R/'example_hcu/ck_tile/19_grouped_gemm/instances';P=Path('/workspace/v7_coverage/p6f')
s=(I/'gemm_pipeline_v7_small_m.hpp').read_text();a=s.index('template <typename Problem, index_t TileN>\nstruct GemmPipelineV7SmallM');b=s.rindex('} // namespace ck_tile') if '} // namespace ck_tile' in s else s.rindex('}   // namespace ck_tile')
s=s[a:b].replace('GemmPipelineV7SmallM','GemmPipelineV7SmallMNoSlc').replace('buffer_load_dwordx4_idxen_lds_t(', 'buffer_load_dwordx4_idxen_lds(')
prefix='''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// P6f limited small-M cache control: unchanged64x128x16, only A SLC removed.
#pragma once
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
namespace ck_tile {
'''
(I/'gemm_pipeline_v7_small_m_noslc.hpp').write_text(prefix+s+'\n} // namespace ck_tile\n#pragma clang diagnostic pop\n')
s=(G/'grouped_gemm_v7_small_experiment_launch.hpp').read_text();a=s.index('namespace v7_small_experiment_launch {');s=s[a:].replace('v7_small_experiment_launch','v7_small_cache_launch')
prefix='''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "grouped_gemm_v7_small_m_launch.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m_noslc.hpp"
struct GemmConfigV7SmallCache : GemmConfigV7SmallM<128>
{
    template <typename Problem>
    using ExperimentPipeline = ck_tile::GemmPipelineV7SmallMNoSlc<Problem,128>;
};
'''
(G/'grouped_gemm_v7_small_cache_launch.hpp').write_text(prefix+s)
cpp=P/'probe_smallcache.cpp';cpp.write_text(f'''#include "{G/'grouped_gemm_v7_small_cache_launch.hpp'}"
extern "C" int probe_smallcache(const void* args, int groups, std::uint64_t tiles, ck_tile::index_t tpg, hipStream_t stream)
{{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return v7_small_cache_launch::launch<GemmConfigV7SmallCache,Row,Row,Row,ck_tile::bf16_t,ck_tile::bf16_t,float,ck_tile::bf16_t>(args,groups,tiles,tpg,stream);
}}
''')
base=json.loads((P/'probe_commands.json').read_text())[0]['command'];base=base[:base.index('-c')]
(P/'probe_smallcache_command.json').write_text(json.dumps(dict(name='smallcache',command=base+['-c',str(cpp),'-o',str(P/'probe_smallcache.o')]),indent=2))
(P/'small_cache_control_plan.json').write_text(json.dumps(dict(at='before any small-M NoSLC GPU measurements',ids=[620,621,76,358],matched_baseline=5003,candidate=5017,geometry=[64,128,16],factor='only NN A SLC modifier',reason='For620/621 A repeated across eight N tiles; possible extra A HBM traffic despite small M. Native families have no SLC. This remains a hypothesis until matched timing/profile.',scope='four representatives only; no general adoption or cache/geometry sweep'),indent=2))
print('SMALL_CACHE_CONTROL_PREPARED four_shapes_only=true')
