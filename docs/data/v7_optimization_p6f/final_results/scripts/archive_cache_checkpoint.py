"""Small, immutable first cache checkpoint; exclude binaries and vendor perf/ISA."""
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
import runtime as r

out=r.REPO/'docs/data/v7_optimization_p6f/cache_checkpoint'
out.mkdir(parents=True,exist_ok=True)
names=['session.json','initial_manifest.json','cache_build_manifest.json','cache_comparisons.json',
       'cache_comparisons.csv','cache_factor_summary.json','native_cache_reference_summary.json',
       'selector_cache_audit.json','cache_isa_audit.json','abi_audit.json',
       'cache_accuracy_plan.json','cache_perf_plan.json','cache_native_plan.json','regression_plan.json',
       'cache_profiles_frozen.json','cache_grid_domain_predeclared.json',
       'runtime.py','health.py','campaign.py','build.py','selector_check.py','selector_cache_cpu.cpp',
       'isa_audit.py','abi_audit.py','analyze_cache.py','analyze_profiles.py',
       'native_replay.py','prepare_native.py','prepare_cache_profiles.py','add_cache_walk.py','profiles.json']
for name in names:
    source=r.ROOT/name
    assert source.is_file(),source
    shutil.copy2(source,out/name)
    if source.suffix=='.csv':
        (out/name).write_bytes(source.read_bytes().replace(b'\r\n',b'\n'))
raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
jobs=[x for x in raw if x['key'].startswith(('e0_regression','e1_regression','e1_cache_native',
    'e1_acc_cache','e1_cache_r')) or x['key'] in ['e1_cache_profile_base1',
    'e1_cache_profile_noslc_m1','e1_cache_profile_base2','e1_cache_profile_noslc_m5','e1_cache_profile_slc_m5']]
assert sum(x.get('ok',False) and x.get('kind')=='perf' for x in jobs)==120
with gzip.open(out/'jobs.jsonl.gz','wt') as f:
    for x in jobs:f.write(json.dumps(x)+'\n')
logs={x['key']:Path(x['path'],'run.log').read_text() for x in jobs if x.get('path') and Path(x['path'],'run.log').exists()}
with gzip.open(out/'logs.json.gz','wt') as f:json.dump(logs,f)
for x in jobs:
    for name,digest in x.get('source_scripts_sha256',{}).items():
        source=r.ROOT/'scripts'/(digest+'_'+name)
        assert source.is_file() and r.sha(source)==digest
        target=out/'scripts'/source.name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
acc=[x for x in jobs if x.get('ok') and x.get('kind')=='accuracy']
summary=dict(at=r.stamp(),benchmark_sha256=r.sha(r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'),
    accuracy_jobs=len(acc),accuracy_comparisons=sum(len(x['comparisons']) for x in acc),
    accuracy_results=[dict(key=x['key'],comparisons=x['comparisons']) for x in acc],
    formal_perf_jobs=120,comparisons=20,default_off=True,
    protected_diff_md5=hashlib.md5(subprocess.check_output(['git','diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'],cwd=r.REPO)).hexdigest(),
    exclusion='e1_cache_profile_slc_m5 returned rc0 but service transition; ok=false; raw preserved',
    scope='Four factor-study shapes; 12-case attribute validation is a later checkpoint; no1193 GPU pass')
assert summary['protected_diff_md5']=='4c7f49a5c7592767d53195a713db6b8a'
(out/'checkpoint.json').write_text(json.dumps(summary,indent=2))
manifest=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('CACHE_CHECKPOINT_ARCHIVED',len(manifest),'files',sum(x['bytes'] for x in manifest),'bytes')
