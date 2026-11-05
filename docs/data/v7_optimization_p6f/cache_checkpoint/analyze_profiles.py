"""Decode actual dispatch geometry and counters; profiler times are diagnostic."""
import importlib.util,json,sys
from pathlib import Path
import runtime as r
sys.path.insert(0,'/workspace/v7_coverage/p6d')
spec=importlib.util.spec_from_file_location('p6f_profile_metrics','/workspace/v7_coverage/p6d/analyze.py');metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
rows=[];raw=[]
for job in [json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]:
    if job.get('kind')!='profile' or not job.get('ok'):continue
    assert len(job['perf_files'])==1
    source=job['perf_files'][0];dispatches=metrics.dispatches(source);geometry=metrics.dispatch_info(source)
    assert dispatches and len(geometry)==len(dispatches)
    shape=tuple(job['shape'][k] for k in ['M','N','K','G']);tile=tuple(job['logical_tile'])
    derived=metrics.p6.metrics(dispatches[-1],shape,tile)
    assert derived['mmop_count_matches_padded'],(job['key'],derived)
    rows.append(dict(key=job['key'],id=job['id'],variant=job.get('variant'),physical_gpu=job['physical_gpu'],epoch=job['epoch'],binary_sha256=job['binary_sha256'],source=source,logical_tile=tile,actual_dispatches=geometry,**derived,note='Counter-derived timing/ratios use existing hardware model, not formal CK time; scratch and dynamic LDS are from actual dispatch'))
    raw.append(dict(key=job['key'],source=source,dispatches=dispatches))
(r.ROOT/'profiles.json').write_text(json.dumps(rows,indent=2));(r.ROOT/'profile_raw_counters.json').write_text(json.dumps(raw,indent=2))
print('PROFILE_ANALYSIS',len(rows))
for x in rows:print(x['key'],'HBM_MB',round(x['hbm_read_MB'],2),'L2_pct',round(x['l2_hit_pct'],2),'latency',round(x['tcp_tcc_read_latency'],1),'geometry',x['actual_dispatches'][-1])
