"""Preserve first frozen plans; add complete epoch-2 contrasts after service event."""
import json
import campaign as c
import runtime as r

p = r.ROOT
old = json.loads((p/'cache_grid_expansion_frozen.json').read_text())
jobs = list(old['jobs'])
case = c.cases()[781]
for rnd in (1, 2, 3):
    pair = f'cache_grid_rec2_r{rnd}_id781'
    for arm, walk, hint, instance in [('control',1,0,5000),('candidate',5,1,5012)]:
        env = dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK=str(walk),
                   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=str(hint))
        spec = c.bench(pair+'_'+arm,case,instance,env,kind='perf')
        spec.update(task=2,arm=arm,round=rnd,pair=pair,variant='cache_grid',
                    baseline_scheme='fresh original5000; whole contrast repeated after ID781 profiler service event')
        jobs.append(spec)
out = dict(old,at=r.stamp(),jobs=jobs,
           reuse_primary_comparisons=[x for x in old['reuse_primary_comparisons'] if x['id']!=781],
           recovery_note='ID781 full three-round A/B repeated in epoch2; earlier clean epoch1 factor study preserved; no cross-epoch pairing')
target=p/'cache_grid_expansion_rec2_frozen.json'
assert not target.exists();target.write_text(json.dumps(out,indent=2))
profiles=json.loads((p/'cache_profiles_frozen.json').read_text())
selected=[]
for variant in ['base3','noslc_m5','slc_m5','noslc_m8','native']:
    spec=dict(next(x for x in profiles['jobs'] if x['variant']==variant))
    spec['key']='cache_profile_rec2_'+variant
    selected.append(spec)
target=p/'cache_profiles_rec2_frozen.json'
assert not target.exists();target.write_text(json.dumps(dict(at=r.stamp(),jobs=selected,
    note='Complete epoch2 baseline/candidate/Native controls; failed epoch1 SLC-M5 excluded'),indent=2))
print('REC2_PLANS_FROZEN',len(jobs),len(selected))
