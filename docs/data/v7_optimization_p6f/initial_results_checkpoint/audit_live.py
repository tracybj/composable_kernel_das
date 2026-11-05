"""Read-only measurement, health, shape, protection and pairing audit."""
import csv,hashlib,json,subprocess
from pathlib import Path
import runtime as r

raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
gates=[json.loads(x) for x in (r.ROOT/'gate.jsonl').read_text().splitlines()]
gate_by={x['key']:x for x in gates}
valid=[x for x in raw if x.get('ok')]
errors=[]
for x in valid:
    g=gate_by[x['key']]
    assert g['rc']==0 and 'MKFD' not in g['output'].upper() and g['hcu']=='0.0%'
    assert g['mem_available_kib']>=64*1024*1024
    assert g['physical_gpu']==x['physical_gpu']==4
    assert x['environment']['HIP_VISIBLE_DEVICES']==x['environment']['CUDA_VISIBLE_DEVICES']=='4'
    assert x['rc']==0 and not x['kernel_service_events']
    log=Path(x['path'],'run.log').read_text()
    assert 'MKFD' not in log.upper()
    if x.get('kind')=='perf':
        for opt in ['--warmup=10','--samples=30','--include-args-build=0','--backend=ck']:
            assert opt in x['command'],(x['key'],opt)
        assert x['selected']==x['expected']
        assert x['result']['median_ms']>0 and x['result']['cv']>=0
    if x.get('kind')=='accuracy':
        assert '--backend=both' in x['command']
        assert x['comparisons'] and all(int(t[1])>0 and int(t[2])==0 for t in x['comparisons'])
    if x.get('id',-1)>0 and x.get('shape'):
        i=x['id'];shape=next(row for row in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig')) if int(row['ID'])==i)
        assert all(x['shape'][k]==int(shape[k]) for k in ['M','N','K','G'])
comparisons=json.loads((r.ROOT/'all_pair_generations.json').read_text()) if (r.ROOT/'all_pair_generations.json').exists() else []
bykey={x['key']:x for x in valid}
for row in comparisons:
    jobs=[bykey[k] for k in row['source_keys']]
    assert len(jobs)==6 and len({x['epoch'] for x in jobs})==1
    assert len({x['binary_sha256'] for x in jobs})==1
    assert len({x['physical_gpu'] for x in jobs})==1
    for rd in (1,2,3):
        a=next(x for x in jobs if x['round']==rd and x['arm']=='control')
        b=next(x for x in jobs if x['round']==rd and x['arm']=='candidate')
        assert a['pair']==b['pair'] and a['finished_at']<=b['at']
        assert not any(a['at']<z['at']<b['at'] for z in raw if z['key'] not in [a['key'],b['key']]),'Non-adjacent formal arms'
for path in ['/workspace/shapes_1178.csv','/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv']:
    assert r.sha(path)=='5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
diff=subprocess.check_output(['git','diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'],cwd=r.REPO)
assert hashlib.md5(diff).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
assert 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp' not in subprocess.check_output(['git','diff','--cached','--name-only'],cwd=r.REPO,text=True).splitlines()
acc=[x for x in valid if x.get('kind')=='accuracy']
record=dict(at=r.stamp(),jobs_attempted=len(raw),jobs_accepted=len(valid),invalid_keys=[x['key'] for x in raw if not x.get('ok')],
    performance_jobs=sum(x.get('kind')=='perf' for x in valid),complete_pair_generations=len(comparisons),
    accuracy_jobs=len(acc),whole_output_comparisons=sum(len(x['comparisons']) for x in acc),
    whole_output_elements=sum(int(t[1]) for x in acc for t in x['comparisons']),
    accuracy_max_abs_values=sorted({t[3] for x in acc for t in x['comparisons']}),
    physical_GPU=4,same_epoch_pairs=True,adjacent_arms=True,shape_hashes_unchanged=True,protected_diff_unchanged_unstaged=True,
    recovery_total=json.loads((r.ROOT/'recovery_total.json').read_text()),
    no_full_gpu_suite=True,note='Audit of completed accepted records only; live/unfinished plans remain pending, not completion of P6f')
(r.ROOT/'live_audit.json').write_text(json.dumps(record,indent=2));print('LIVE_AUDIT_PASS',json.dumps(record))
