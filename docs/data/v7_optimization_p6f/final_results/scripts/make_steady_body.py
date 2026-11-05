"""Separate steady and five-stage drain using CK-owned pipeline bodies."""
from pathlib import Path
import json
R=Path('/workspace/composable_kernel/include/ck_tile/ops/gemm/pipeline')
OUT=Path('/workspace/v7_coverage/p6f')
def generate(k32):
    old='GemmPipelineV7SmallMK32' if k32 else 'GemmPipelineV7SmallM'
    new=old+'Steady'
    filename='gemm_pipeline_v7_small_m'+('_k32' if k32 else '')+'.hpp'
    s=(R/filename).read_text()
    a=s.index('template <typename Problem, index_t TileN>\nstruct '+old)
    b=s.rindex('} // namespace ck_tile') if '} // namespace ck_tile' in s else s.rindex('}   // namespace ck_tile')
    s=s[a:b].replace(old,new)
    s=s.replace('template <int kBank, uint32_t kReadSlot, uint32_t kLoadSlot>', 'template <int kBank, uint32_t kReadSlot, uint32_t kLoadSlot, int Mode>')
    s=s.replace('if((kt + 2) < s.n_stages)','if(Mode == 0 || (kt + 2) < s.n_stages)')
    oldstage='const int kStage   = (kt + G::kPrefetchDist < s.n_stages) ? kt + G::kPrefetchDist : 0;'
    assert s.count(oldstage)==1;s=s.replace(oldstage,'if constexpr(Mode == 0)\n        {\n        const int kStage = kt + G::kPrefetchDist;')
    a=s.index('    // One k-stage.');pre=s[:a];pos=pre.rfind('    }');pre=pre[:pos]+'        } // no global requests in drain modes\n'+pre[pos:];s=pre+s[a:]
    s=s.replace('              bool kBarrier = true,','              bool kBarrier = true,\n              int Mode = 0,')
    s=s.replace('issue_load_and_read<kBankNext, kReadSlot, kLoadSlot>', 'issue_load_and_read<kBankNext, kReadSlot, kLoadSlot, Mode>')
    reads='G::kReadsPerStage' if k32 else 'G::kMTilesPerWave + G::kNTilesPerWave / 2'
    loads='G::kLoadsPerStage' if k32 else '2'
    waitold=f'        v7u::wait_all_buffer_data_arrived<false>({"2 * G::kLoadsPerStage" if k32 else "4"});'
    assert s.count(waitold)==1
    s=s.replace(waitold,f'''        if constexpr(Mode == 0) v7u::wait_all_buffer_data_arrived<false>(2 * {loads});
        else if(kt + 4 < s.n_stages) v7u::wait_all_buffer_data_arrived<false>({loads});
        else v7u::wait_all_buffer_data_arrived<false>(0);''')
    # Prologue stage4 only exists for S>=5. At S=4 the four original
    # loads are enough; do not replace already-consumed slot0 with stage0.
    needle='    {\n        constexpr uint32_t kPreSlot ='
    assert s.count(needle)==1;s=s.replace(needle,'    if(S.n_stages > 4)\n    {\n        constexpr uint32_t kPreSlot =')
    a=s.index('    // The k loop.');s=s[:a]
    s+='''    // Steady mode has no per-stage load/read bounds. Tail mode is selected
    // by one phase switch, then bounded fallthrough. Avoid both a five-body
    // expansion per arm and loop-carried bank phis across a drain switch.
    // Keep the original divergent wave-group arm and complete LDS waits.
    int kt = 0;
'''
    for arm,group in enumerate(['true','false']):
        s+=('    if(S.wid < (int)G::kWaveGroup)\n' if arm==0 else '    else\n')+'    {\n'
        s+='''
#pragma unroll 1
            for(bool go = (kt + (int)G::kPhaseCycle + 5 <= S.n_stages); go;)
            {
'''
        for pos in range(12):s+=f'                stage_step<{pos}, {group}, {"false" if pos==11 else "true"}, 0>(S, acc, a_bank, b_bank, kt + {pos});\n'
        s+='''                kt += (int)G::kPhaseCycle;
                go = (kt + (int)G::kPhaseCycle + 5 <= S.n_stages);
                wait_all_warp_arrived();
            }
'''
        for pos in range(11):s+=f'            if(kt + 5 < S.n_stages) {{ stage_step<{pos}, {group}, true, 0>(S, acc, a_bank, b_bank, kt); ++kt; }}\n'
        s+='''
            switch(kt % (int)G::kPhaseCycle)
            {
'''
        for phase in range(12):
            s+=f'            case {phase}:\n'
            s+=f'                stage_step<{phase}, {group}, true, -1>(S, acc, a_bank, b_bank, kt);\n'
            s+='                if(++kt == S.n_stages) break;\n'
            if phase<11:s+='                [[fallthrough]];\n'
            else:
                for wrap in range(4):
                    s+=f'                stage_step<{wrap}, {group}, true, -1>(S, acc, a_bank, b_bank, kt);\n'
                    s+='                if(++kt == S.n_stages) break;\n'
                s+='                break;\n'
        s+='''            }
    }
'''
    s+='}\n\n} // namespace ck_tile\n#pragma clang diagnostic pop\n'
    out='gemm_pipeline_v7_small_m'+('_k32' if k32 else '_k16')+'_steady.hpp'
    prefix=f'''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// P6f independent steady/drain: no redundant global requests, unchanged
// physical panels, operand banks, wave order and LDS completion barriers.
#pragma once
#include "ck_tile/ops/gemm/pipeline/{filename}"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
namespace ck_tile {{
'''
    (R/out).write_text(prefix+s)
    print('STEADY_BODY_CREATED',out)

def model():
    checked=0
    for S in range(4,1025):
        ring={};banks={};pending=[];loads=[];reads=[];computed=[]
        def load(t):
            assert 0<=t<S
            slot=t%4
            prior=ring.get(slot)
            assert prior is None or prior in reads, ('overwrite before read',S,t,prior)
            ring[slot]=t;pending.append(t);loads.append(t)
        def wait(left):
            while len(pending)>left:pending.pop(0)
        def read(t):
            assert ring[t%4]==t and t not in pending,('unready',S,t,pending,ring)
            banks[t%3]=t;reads.append(t)
        for t in range(4):load(t)
        wait(3);read(0);wait(2);read(1);wait(1)
        if S>4:load(4)
        for t in range(S):
            assert banks[t%3]==t,('wrong operand bank',S,t,banks)
            computed.append(t)
            if t+2<S:read(t+2)
            if t+5<S:load(t+5);wait(2)
            elif t+4<S:wait(1)
            else:wait(0)
        assert loads==list(range(S)) and sorted(reads)==list(range(S)) and computed==list(range(S))
        checked+=S
    (OUT/'steady_schedule_model.json').write_text(json.dumps(dict(stage_counts_tested=[4,1024],stages_checked=checked,loads_per_K_stage=1,reads_per_K_stage=1,compute_per_K_stage=1,ring_slots=4,operand_banks=3,scope='CPU ordered request/lifetime model; not GPU compiler/synchronization proof'),indent=2))
    print('STEADY_LIFETIME_MODEL_PASS stages='+str(checked))
if __name__=='__main__':
    model();generate(True);generate(False)
