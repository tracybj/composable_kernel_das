"""Conditional H3 scheduling follow-up: move only private MMAC across ready.

No provider/source/selector integration, compiler invocation or GPU operation.
This proves operand-bank readiness and unchanged per-wave K accumulation order,
not global memory correctness, target ISA scheduling, overlap or performance.
The existing resource producer/read/ready/barrier schedule remains unchanged.
"""
from pathlib import Path
import hashlib,json

def simulate(stages,waves):
    assert stages>=2 and waves in [4,8]
    state=[{'bank':[[0,True],[1,True],[None,False]],'pending':[], 'calcs':[], 'barriers':0} for _ in range(waves)]
    for t in range(stages):
        # Both groups have identical iteration/barrier counts. MMAC first uses
        # operands retired by the previous ready, or by the initial prologue.
        for wave,s in enumerate(state):
            if wave<waves//2:
                assert s['bank'][t%3]==[t,True],(stages,wave,t,s)
                s['calcs'].append(t)
            # Existing lgkmcnt(0) is retained: previous read(t+1) must retire
            # before a shared producer slot can be reused. No loose wait here.
            for bank,stage in s['pending']:
                assert s['bank'][bank]==[stage,False]
                s['bank'][bank][1]=True
            s['pending']=[];s['barriers']+=1
        # Global VMEM readiness and visibility are delegated to the unchanged,
        # separately verified resource schedule. This model does not relax it.
        for wave,s in enumerate(state):
            if t+2<stages:
                target=(t+2)%3
                assert target!=t%3
                assert s['bank'][target][1] or s['bank'][target][0] is None
                s['bank'][target]=[t+2,False];s['pending'].append((target,t+2))
            if wave>=waves//2:
                assert s['bank'][t%3]==[t,True],(stages,wave,t,s)
                s['calcs'].append(t)
    for s in state:assert s['calcs']==list(range(stages)) and s['barriers']==stages
    return stages*waves

def main():
    schedules=checks=0
    for stages in range(2,1025):
        for waves in [4,8]:checks+=simulate(stages,waves);schedules+=1
    result=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_only=True,GPU=False,source_integrated=False,schedules=schedules,per_wave_stage_checks=checks,
        ok=True,unchanged='Complete-M ownership,resource footprint,producer/read/barrier order,VM/LG thresholds,three operand banks,per-wave K/FMA order and output mapping',
        proposed_change='First half of waves computes current MMAC before ready; other half keeps current MMAC after future read/prefetch. One outer divergent branch, not a per-stage branch.',
        evidence_limits='Only private-operand readiness/alias/order proof. Actual instruction ordering/registers/full precision,resources and formal paired timing remain untested.',
        prerequisites='Finish fixed46 resource pairs and batch8. Proceed only if remaining scheduling/overlap mechanism is supported; no new ownership architecture and no blind parameter sweep.',
        original_deadline_unchanged=True)
    path=Path(__file__).with_name('resource_interleave_cpu_model_result.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print('CONDITIONAL_RESOURCE_INTERLEAVE_CPU_MODEL_PASS',schedules,checks,'no integration/build/GPU',flush=True)

if __name__=='__main__':main()
