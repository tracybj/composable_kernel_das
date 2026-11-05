"""Immutable initial three-direction results, before optional resource-domain outcome."""
import csv,gzip,hashlib,json,shutil,statistics
from pathlib import Path
import runtime as r

out=r.REPO/'docs/data/v7_optimization_p6f/initial_results_checkpoint';out.mkdir(parents=True,exist_ok=True)
pairs=[x for x in json.loads((r.ROOT/'per_shape_results.json').read_text()) if x['variant']!='k32_resource']
assert len(pairs)==129
profiles=json.loads((r.ROOT/'profiles.json').read_text());assert len(profiles)==28
native=json.loads((r.ROOT/'native_diagnostic_summary.json').read_text())
for name,data in [('per_shape_results.json',pairs),('profiles.json',profiles),('native_diagnostic_summary.json',native)]:
    (out/name).write_text(json.dumps(data,indent=2))
with (out/'per_shape_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(pairs[0]),lineterminator='\n');w.writeheader()
    for row in pairs:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
regressions=[x for x in pairs if x['paired_delta_pct']>2]
(out/'all_regressions.json').write_text(json.dumps(regressions,indent=2))
names=['analyze_all_pairs.py','analyze_profiles.py','analyze_native.py','audit_live.py','make_incremental_views.py',
       'audit_incremental_views.py','runtime.py','health.py','campaign.py','phase_initial_followups.py',
       'extra_accuracy_frozen.json','isolation_accuracy_frozen.json','isolation_perf_frozen.json',
       'isolation_profiles_filterfix_frozen.json','decode_profiles_filterfix_frozen.json',
       'queue_controls.jsonl','execution_schedules.jsonl','recovery_total.json','environment_epoch.json']
for name in names:shutil.copy2(r.ROOT/name,out/name)
raw=[json.loads(s) for s in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
jobs=[x for x in raw if x.get('variant')!='k32_resource']
with gzip.open(out/'jobs.jsonl.gz','wt') as f:
    for x in jobs:f.write(json.dumps(x)+'\n')
with gzip.open(out/'logs.json.gz','wt') as f:json.dump({x['key']:Path(x['path'],'run.log').read_text() for x in jobs},f)
for name in ['recovery.jsonl','gate.jsonl','kernel_checks.jsonl','decisions.jsonl']:
    with gzip.open(out/(name+'.gz'),'wb') as f:f.write((r.ROOT/name).read_bytes())
for job in jobs:
    for name,digest in job.get('source_scripts_sha256',{}).items():
        source=r.ROOT/'scripts'/(digest+'_'+name);target=out/'scripts'/source.name;target.parent.mkdir(exist_ok=True)
        assert r.sha(source)==digest;shutil.copy2(source,target)
accuracy=[x for x in jobs if x.get('ok') and x.get('kind')=='accuracy']
summary=dict(at=r.stamp(),formal_jobs=774,complete_contrasts=129,accepted_profiles=28,
    accuracy_jobs=len(accuracy),whole_output_comparisons=sum(len(x['comparisons']) for x in accuracy),
    whole_output_elements=sum(int(t[1]) for x in accuracy for t in x['comparisons']),
    failed_keys=[x['key'] for x in jobs if not x.get('ok')],
    tail256=dict(shapes=16,all_beat_frozen_BLAS=True,R_range=[min(x['R_after_frozen'] for x in pairs if x['variant']=='tail256'),max(x['R_after_frozen'] for x in pairs if x['variant']=='tail256')]),
    adopted_flags=['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1 plus CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL=2',
                   'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1 plus CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID=1'],
    decode_new_refactor='No broad adoption: same-geometry changes not beyond measured noise',
    pending='159-case existing-K32 resource domain and optional explicit multi-launch diagnostic; these are separate from this completed initial checkpoint',
    default_off=True,no1193_gpu_replay=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
manifest=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('INITIAL_RESULTS_ARCHIVED',len(manifest),'files',sum(x['bytes'] for x in manifest),'bytes',summary)
