from pathlib import Path
import json
ROOT=Path('/workspace/v7_coverage/p6f');G=Path('/workspace/composable_kernel/example_hcu/ck_tile/19_grouped_gemm/instances')
s=(G/'grouped_gemm_v7_small_experiment_launch.hpp').read_text();a=s.index('namespace v7_small_experiment_launch {');s=s[a:].replace('v7_small_experiment_launch','v7_steady_launch')
prefix='''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "grouped_gemm_v7_small_m_launch.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m_k32_steady.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_small_m_k16_steady.hpp"
template <int Kind>
struct GemmConfigV7Steady : GemmConfigV7SmallM<128>
{
    static constexpr int kBlockPerCu = Kind == 1 ? 1 : 2;
    template <typename Problem>
    using ExperimentPipeline = std::conditional_t<Kind == 1,
        ck_tile::GemmPipelineV7SmallMK32Steady<Problem,128>,
        ck_tile::GemmPipelineV7SmallMSteady<Problem,128>>;
};
'''
(G/'grouped_gemm_v7_steady_launch.hpp').write_text(prefix+s)
base=json.loads(Path('/workspace/v7_coverage/p6e/probe_build_commands.json').read_text())[0]['command'];base=base[:base.index('-c')]
cmds=[]
for name,header,namespace,config in [('tail','grouped_gemm_v7_tail_launch.hpp','v7_tail_launch','GemmConfigV7Tail128'),('k32steady','grouped_gemm_v7_steady_launch.hpp','v7_steady_launch','GemmConfigV7Steady<1>'),('k16steady','grouped_gemm_v7_steady_launch.hpp','v7_steady_launch','GemmConfigV7Steady<0>')]:
    cpp=ROOT/('probe_'+name+'.cpp')
    cpp.write_text(f'''#include "{G/header}"
extern "C" int probe_{name}(const void* args, int groups, std::uint64_t tiles, ck_tile::index_t tpg, hipStream_t stream)
{{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return {namespace}::launch<{config},Row,Row,Row,ck_tile::bf16_t,ck_tile::bf16_t,float,ck_tile::bf16_t>(args,groups,tiles,tpg,stream);
}}
''')
    cmds.append(dict(name=name,command=base+['-c',str(cpp),'-o',str(ROOT/('probe_'+name+'.o'))]))
(ROOT/'probe_commands.json').write_text(json.dumps(cmds,indent=2))
print('PROBE_COMMANDS_PREPARED',len(cmds))
