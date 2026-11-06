"""Read profiler counters; preserve inferred-format caveat and actual dispatch geometry."""
import importlib.util,sys,json
from pathlib import Path
import runtime as r
sys.path.insert(0,'/workspace/v7_coverage/p6d')
s=importlib.util.spec_from_file_location('prior_metrics','/workspace/v7_coverage/p6d/analyze.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for j in map(json.loads,(r.ROOT/'jobs.jsonl').read_text().splitlines()):
 if j.get('kind')!='profile' or not j.get('ok'):continue
 f=j['perf_files'][0];ds=m.dispatches(f);geom=m.dispatch_info(f);assert ds and len(geom)==len(ds)
 out=m.p6.metrics(ds[-1],tuple(j['shape'][k] for k in ['M','N','K','G']),tuple(j['logical_tile']));assert out['mmop_count_matches_padded']
 x=dict(key=j['key'],id=j['id'],variant=j['variant'],binary_sha256=j['binary_sha256'],epoch=j['epoch'],geometry=geom,metrics=out,raw=ds[-1],note='xprof dispatch layout locally inferred; profile timing not used as formal CK timing');rows.append(x)
 print(j['id'],j['variant'],'geometry',geom[-1]);print({k:out[k] for k in ['hbm_read_MB','l2_hit_pct','tcp_tcc_read_latency','dram_credit_stall_frac','lds_bank_conflict_frac','waves','mmop_insts']})
 c=ds[-1];print({k:c[k]/c['SQ_WAVES'] for k in ['SQ_INSTS_VALU','SQ_INSTS_VMEM','SQ_INSTS_LDS','SQ_INSTS_SALU','SQ_INSTS_BRANCH'] if k in c});print({k:v for k,v in c.items() if any(t in k for t in ['WAIT','STALL'])})
(r.ROOT/'profile_results.json').write_text(json.dumps(rows,indent=2));print('PROFILE_COUNTER_AUDIT_PASS',len(rows))
