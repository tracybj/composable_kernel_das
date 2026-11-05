"""Archive completed new-instance precision and official build evidence."""
import gzip,hashlib,json,shutil
from pathlib import Path
import runtime as r

binary=json.loads((r.ROOT/'impl_build_manifest.json').read_text())['benchmark_sha256']
raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
plans=['tail_accuracy_both_tiles_frozen.json','decode_accuracy_with_baselines_frozen.json','small_cache_accuracy_frozen.json']
expected={x['key'] for plan in plans for x in json.loads((r.ROOT/plan).read_text())['jobs']}
accepted={x['logical_key']:x for x in raw if x.get('ok') and x.get('kind')=='accuracy' and x.get('campaign_binary_sha256')==binary}
assert len(expected)==246 and expected<=set(accepted),(len(expected-set(accepted)),sorted(expected-set(accepted))[:4])
selected=[accepted[k] for k in sorted(expected)]
out=r.REPO/'docs/data/v7_optimization_p6f/implementation_checkpoint';out.mkdir(parents=True,exist_ok=True)
names=['impl_build_manifest.json','impl_source_manifest.json','impl_isa_audit.json','impl_static_compile_resources.json',
       'selector_impl_audit.json','selector_cache_audit.json','abi_audit.json','impl_profile_symbol_audit.json',
       'tail_address_model.json','tailwide_b64_model.json','tail_crop_model.json','steady_schedule_model.json',
       'steady_phase_fallthrough_model.json','prototype_resources_current.json','steady_design.md','measurement_scope_notes.md',
       'selector_tail_steady_cpu.cpp','selector_cache_grid_cpu.cpp','selector_check.py','selector_cache_cpu.cpp',
       'isa_impl_audit.py','verify_impl_profile_plans.py','integrate_tail_steady.py','integrate_cache_grid.py',
       'prepare_impl_perf.py','prepare_implementation_plans.py','phase_impl.py',
       'tail_accuracy_both_tiles_frozen.json','decode_accuracy_with_baselines_frozen.json','small_cache_accuracy_frozen.json',
       'tail_perf_frozen.json','decode_representative_perf_frozen.json','small_cache_perf_frozen.json',
       'cache_policy_verification_frozen.json','tail_profiles_verified_frozen.json','decode_profiles_verified_frozen.json',
       'extra_accuracy_frozen.json','runtime.py','health.py','campaign.py']
for name in names:shutil.copy2(r.ROOT/name,out/name)
for name in ['build_impl.log','build_impl_static.log']:
    with gzip.open(out/(name+'.gz'),'wb') as f:f.write((r.ROOT/name).read_bytes())
jobs=[x for x in raw if x.get('campaign_binary_sha256')==binary and x.get('kind') in ['accuracy','regression']]
with gzip.open(out/'jobs.jsonl.gz','wt') as f:
    for x in jobs:f.write(json.dumps(x)+'\n')
with gzip.open(out/'logs.json.gz','wt') as f:json.dump({x['key']:Path(x['path'],'run.log').read_text() for x in jobs},f)
summary=dict(at=r.stamp(),benchmark_sha256=binary,accuracy_jobs=246,
    whole_output_comparisons=sum(len(x['comparisons']) for x in selected),
    checked_elements=sum(int(t[1]) for x in selected for t in x['comparisons']),
    max_abs_values=sorted({t[3] for x in selected for t in x['comparisons']}),
    task_counts={str(t):sum(x['task']==t for x in selected) for t in [1,2,3]},
    accepted_keys=[x['key'] for x in selected],invalid_keys=[x['key'] for x in jobs if not x.get('ok')],
    epochs=sorted({x['epoch'] for x in selected}),physical_gpu=4,
    no_performance_adoption_claim=True,extra_boundary_plan='8 additional corners still pending at this checkpoint')
(out/'precision_checkpoint.json').write_text(json.dumps(summary,indent=2))
manifest=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('IMPL_PRECISION_ARCHIVED',summary['accuracy_jobs'],summary['whole_output_comparisons'],summary['checked_elements'],len(manifest))
