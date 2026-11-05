"""Archive the completed finite cache-domain diagnosis independently of new code."""
import gzip,hashlib,json,shutil
import runtime as r

out=r.REPO/'docs/data/v7_optimization_p6f/cache_domain_checkpoint';out.mkdir(parents=True,exist_ok=True)
names=['cache_grid_results.json','cache_grid_summary.json','cache_grid_domain_predeclared.json',
       'cache_grid_expansion_frozen.json','cache_grid_expansion_rec2_frozen.json',
       'cache_grid_policy_domain_refined.json','cache_policy_verification_frozen.json',
       'cache_profiles_rec2_frozen.json','profiles.json','invalidations.json','live_audit.json',
       'environment_epoch.json','recovery_total.json','checkpoint_cache_final_diagnostics.json',
       'analyze_all_pairs.py','analyze_cache_grid.py','analyze_profiles.py','audit_live.py',
       'prepare_cache_rec2.py','phase_cache_rec2.py','integrate_cache_grid.py','health.py','runtime.py']
for name in names:shutil.copy2(r.ROOT/name,out/name)
for name in ['jobs.jsonl','gate.jsonl','kernel_checks.jsonl','recovery.jsonl','waits.jsonl','decisions.jsonl']:
    with gzip.open(out/(name+'.gz'),'wb') as f:f.write((r.ROOT/name).read_bytes())
jobs=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
logs={x['key']:open(x['path']+'/run.log').read() for x in jobs}
for row in [json.loads(x) for x in (r.ROOT/'recovery.jsonl').read_text().splitlines()]:logs[row['key']]=open(row['path']+'/run.log').read()
with gzip.open(out/'logs.json.gz','wt') as f:json.dump(logs,f)
for x in jobs:
    for name,digest in x.get('source_scripts_sha256',{}).items():
        source=r.ROOT/'scripts'/(digest+'_'+name);target=out/'scripts'/source.name
        assert r.sha(source)==digest;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
manifest=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json']
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('CACHE_DOMAIN_ARCHIVED',len(manifest),sum(x['bytes'] for x in manifest))
