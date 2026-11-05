"""Final small source/derived evidence archive. Vendor code objects stay onsite."""
import gzip, hashlib, json, shutil
from pathlib import Path
import runtime as r

out=r.REPO/'docs/data/v7_optimization_p6f/final_results'
assert not out.exists(),'Final archive is immutable; review before creating'
out.mkdir(parents=True)
names=[
    'session.json','final_status.json','final_audit.json','final_summary.json','report.md',
    'progress_final.md',
    'per_shape_results.json','per_shape_results.csv','all_pair_generations.json','partial_pair_groups.json',
    'all_regressions.json','k32_resource_summary.json','new15_results.json',
    'measurement_exclusions.json',
    'profiles.json','native_diagnostic_summary.json','launch_scope_results.json',
    'policy_overlay_incremental_1193.json','policy_overlay_incremental_1193.csv',
    'historical_envelope_incremental_1193.json','historical_envelope_incremental_1193.csv',
    'incremental_summary.json','incremental_view_audit.json','live_audit.json',
    'abi_audit.json','selector_impl_audit.json','impl_isa_audit.json','benchmark_only_shader_preservation.json',
    'launch_diagnostic_install.json','recovery_total.json','environment_epoch.json',
    'k32_resource_domain_predeclared.json','next_hypotheses_cpu_only.md',
    'selector_incremental_probe.cpp','selector_incremental_checks.csv',
]
names += [x.name for x in r.ROOT.glob('*_frozen.json')]
names += [x.name for x in r.ROOT.glob('build_*_command.json')]
for name in sorted(set(names)):
    source=r.ROOT/name
    if source.exists():shutil.copy2(source,out/name)
for source in sorted(r.ROOT.glob('*.py')):
    if source.name.startswith(('campaign_pre_','pre_launch_','campaign_ordered_pending')):continue
    folder=out/'scripts';folder.mkdir(exist_ok=True);shutil.copy2(source,folder/source.name)
raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
with gzip.open(out/'jobs.jsonl.gz','wt') as f:
    for x in raw:f.write(json.dumps(x)+'\n')
with gzip.open(out/'run_logs.json.gz','wt') as f:
    json.dump({x['key']:Path(x['path'],'run.log').read_text() for x in raw},f)
for name in ['gate.jsonl','kernel_checks.jsonl','recovery.jsonl','waits.jsonl','decisions.jsonl',
             'queue_controls.jsonl','execution_schedules.jsonl','builds.jsonl','impl_steps.jsonl',
             'launch_scope_steps.jsonl']:
    source=r.ROOT/name
    if source.exists():
        with gzip.open(out/(name+'.gz'),'wb') as f:f.write(source.read_bytes())
for source in sorted(r.ROOT.glob('build_*.log')):
    folder=out/'build_logs';folder.mkdir(exist_ok=True)
    with gzip.open(folder/(source.name+'.gz'),'wb') as f:f.write(source.read_bytes())
for job in raw:
    for name,digest in job.get('source_scripts_sha256',{}).items():
        source=r.ROOT/'scripts'/(digest+'_'+name)
        target=out/'source_versions'/source.name;target.parent.mkdir(exist_ok=True)
        assert r.sha(source)==digest;shutil.copy2(source,target)
for source in sorted(r.ROOT.glob('recovery_*_resolved.json')):
    folder=out/'resolved_events';folder.mkdir(exist_ok=True);shutil.copy2(source,folder/source.name)
# Health probes are separate known-correct binaries: preserve their commands,
# hashes and small logs, never the binary or vendor ISA/perf artifacts.
for line in (r.ROOT/'recovery.jsonl').read_text().splitlines():
    x=json.loads(line);source=Path(x['path'])/'run.log'
    folder=out/'recovery_logs';folder.mkdir(exist_ok=True)
    if source.exists():shutil.copy2(source,folder/(x['key']+'.log'))
manifest=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
          for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('FINAL_ARCHIVED',len(manifest),'files',sum(x['bytes'] for x in manifest),'bytes')
