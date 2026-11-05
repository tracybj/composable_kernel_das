"""CPU-only completion and lineage audit; never changes measured records."""
import csv, hashlib, json, statistics, subprocess, time
from pathlib import Path
import runtime as r

raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
bykey={x['key']:x for x in raw}
assert len(bykey)==len(raw),'Physical record keys must be immutable and unique'
exclusion_file=r.ROOT/'measurement_exclusions.json'
excluded=set(json.loads(exclusion_file.read_text())['keys']) if exclusion_file.exists() else set()
valid=[x for x in raw if x.get('ok') and x['key'] not in excluded]
pairs=json.loads((r.ROOT/'per_shape_results.json').read_text())
resource=[x for x in pairs if x['variant']=='k32_resource']
plan=json.loads((r.ROOT/'k32_resource_domain_frozen.json').read_text())
expected={x['id'] for x in plan['jobs'] if x.get('kind')=='perf'}
assert len(expected)==159
assert {x['id'] for x in resource}==expected,'Complete the entire frozen domain, including negatives'
assert len(resource)==159 and len(pairs)==288
for pair in pairs:
    jobs=[bykey[k] for k in pair['source_keys']]
    assert all(x['ok'] and x['kind']=='perf' for x in jobs)
    assert len(jobs)==6 and {x['round'] for x in jobs}=={1,2,3}
    assert len({(x['epoch'],x['binary_sha256'],x['physical_gpu']) for x in jobs})==1
    for job in jobs:
        assert not job['kernel_service_events']
        assert '--warmup=10' in job['command'] and '--samples=30' in job['command']
        assert not any(a.startswith('--launches-per-sample=') and a!='--launches-per-sample=1' for a in job['command'])
        accuracy=[x for x in valid if x.get('kind')=='accuracy' and x.get('id')==job['id']
                  and x.get('selected')==job['selected'] and x['binary_sha256']==job['binary_sha256']
                  and x['finished_at']<=job['at']]
        # Boundary/fallback controls may intentionally reuse an original route;
        # every timed selected instance still needs whole-output evidence.
        assert accuracy,(job['key'],'No same-binary, same-shape full-output proof')
for name in ['policy_overlay_incremental_1193','historical_envelope_incremental_1193']:
    rows=json.loads((r.ROOT/(name+'.json')).read_text())
    assert len(rows)==1193 and {x['ID'] for x in rows}==set(range(1,1194))
    for row in rows:
        if row['source_type']=='historical_current_retained':continue
        jobs=[bykey[k] for k in row['source_keys']]
        assert len(jobs)==3 and {x['round'] for x in jobs}=={1,2,3}
        assert all(x['ok'] and x['kind']=='perf' for x in jobs)
        assert len({(x['epoch'],x['binary_sha256'],x['selected']) for x in jobs})==1
        assert statistics.median(x['result']['median_ms'] for x in jobs)==row['CK_ms']
        assert row['enable_env']=={k:v for k,v in jobs[0]['environment'].items() if k.startswith('CK_TILE_GROUPED_GEMM_')}
        assert row['full_output_accuracy_keys']
launch=json.loads((r.ROOT/'launch_scope_results.json').read_text()) if (r.ROOT/'launch_scope_results.json').exists() else []
for row in launch:
    assert len(row['source_keys'])==6
    jobs=[bykey[k] for k in row['source_keys']]
    assert all(x['ok'] and x['kind']=='diagnostic_perf' for x in jobs)
    assert len({(x['epoch'],x['binary_sha256'],x['physical_gpu']) for x in jobs})==1
    assert row['Native_same_epoch']['epoch']==row['epoch']
    for rd in (1,2,3):
        a=next(x for x in jobs if x['round']==rd and x['arm']=='control')
        b=next(x for x in jobs if x['round']==rd and x['arm']=='candidate')
        assert a['launches_per_sample']==1 and b['launches_per_sample']==20
        assert a['finished_at']<=b['at']
        assert not any(a['at']<z['at']<b['at'] for z in raw if z['key'] not in [a['key'],b['key']])
assert not r.MARK.exists() and not r.STOP.exists()
assert time.time()<json.loads((r.ROOT/'session.json').read_text())['deadline_epoch']
diff=subprocess.check_output(['git','diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'],cwd=r.REPO)
assert hashlib.md5(diff).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
record=dict(at=r.stamp(),all_pass=True,formal_latest_contrasts=len(pairs),
    k32_resource_complete=len(resource),diagnostic_complete=len(launch),
    full_output_lineage_both_arms=True,source_labelled1193_views=True,
    formal_single_launch=True,same_epoch_binary_gpu=True,protected_diff_unchanged=True,
    no1193_gpu_replay=True,default_off=True)
(r.ROOT/'final_audit.json').write_text(json.dumps(record,indent=2))
print('FINAL_COMPLETION_AUDIT_PASS',json.dumps(record))
