"""Predeclared all-state cache diagnostics, including repeated original control."""
import json,sys
from pathlib import Path
import campaign as c
import runtime as r
cs=c.cases();i=781;jobs=[]
for name,walk,noslc,num in [('base1',1,0,5000),('noslc_m1',1,1,5005),('base2',1,0,5000),('noslc_m5',5,1,5012),('slc_m5',5,0,5011),('noslc_m8',8,1,5006),('base3',1,0,5000)]:
    e=dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK=str(walk),CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=str(noslc))
    spec=c.bench('cache_profile_'+name,cs[i],num,e,kind='profile');spec['command']=[x.replace('--warmup=1','--warmup=10') for x in spec['command']]
    spec.update(variant=name,logical_tile=[256,256,16],task=2);jobs.append(spec)
case=next(x for x in json.loads((r.ROOT/'cases.json').read_text()) if x['id']==i)
script=r.ROOT/'native_replay.py'
jobs.append(dict(key='cache_profile_native',id=i,shape=cs[i],variant='native',kind='profile',task=2,logical_tile=[256,256,16],kernel_filter=case['kernel_name'],binary=sys.executable,replay_script_sha256=r.sha(script),command=[sys.executable,'-u',str(script),f'--id={i}','--warmup=10','--samples=1','--repeats=1','--validate=0'],env=dict(LD_PRELOAD=r.BLAS)))
path=r.ROOT/'cache_profiles_frozen.json';assert not path.exists()
path.write_text(json.dumps(dict(at=r.stamp(),reason='Observed original-control bimodality; retain all three control captures plus cache/walk factors and frozen Native; warmup10/sample1, analyze all dispatches and last warmed dispatch',jobs=jobs),indent=2))
print('CACHE_PROFILE_PLAN_FROZEN',len(jobs))
