#!/usr/bin/env python3
"""Cache-only interleaved rounds and measured memory traffic; no winner picking."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p6d_metrics','/workspace/v7_coverage/p6d/analyze.py')
p6d=importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0,'/workspace/v7_coverage/p6d')
spec.loader.exec_module(p6d)

def main():
 jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
 good={r['key']:r for r in jobs if r.get('ok')}
 frozen={int(r['ID']):r for r in csv.DictReader(open('/workspace/bf16NN-groupgemm.csv',encoding='utf-8-sig'))}
 times=[];profiles=[];raw=[]
 ids=sorted({r['id'] for r in good.values() if r['key'].startswith('perf_cache_')})
 for i in ids:
  b=[good[f'perf_cache_r{n}_id{i}_base']['result'] for n in [1,2,3] if f'perf_cache_r{n}_id{i}_base' in good]
  a=[good[f'perf_cache_r{n}_id{i}_noslc']['result'] for n in [1,2,3] if f'perf_cache_r{n}_id{i}_noslc' in good]
  if len(a)!=3 or len(b)!=3:continue
  before=statistics.median(x['median_ms'] for x in b);after=statistics.median(x['median_ms'] for x in a)
  times.append(dict(id=i,**{k:int(frozen[i][k]) for k in ['M','N','K','G']},before_ms=before,noslc_ms=after,delta_pct=100*(after/before-1),before_round_ms=[r['median_ms'] for r in b],noslc_round_ms=[r['median_ms'] for r in a],before_cv=[r['cv'] for r in b],noslc_cv=[r['cv'] for r in a],frozen_blas_gpu0_ms=float(frozen[i]['BLAS ms']),R_before=float(frozen[i]['BLAS ms'])/before,R_noslc=float(frozen[i]['BLAS ms'])/after))
 for r in good.values():
  if not r['key'].startswith('profile_cache_'):continue
  ds=p6d.dispatches(r['perf_files'][0]);assert len(ds)==2
  shape=tuple(r['shape'][k] for k in ['M','N','K','G']);metric=p6d.p6.metrics(ds[-1],shape,(256,256,16))
  geometry=p6d.dispatch_info(r['perf_files'][0]);assert len(geometry)==2
  for d,g in zip(ds,geometry):
   assert d['SQ_INSTS_MMOP']==metric['mmop_insts_expected_padded']
   assert d['SQ_WAVES']==metric['ctas']*8 and g['ctas']==metric['ctas']
  profiles.append(dict(id=r['id'],variant=r['variant'],geometry=geometry,source=r['perf_files'][0],**metric))
  raw.append(dict(key=r['key'],dispatches=ds,source=r['perf_files'][0]))
 for name,value in [('cache_times.json',times),('cache_profiles.json',profiles),('cache_raw_counters.json',raw)]:
  (ROOT/name).write_text(json.dumps(value,indent=2))
 if times:
  with (ROOT/'cache_times.csv').open('w') as f:
   fields=[k for k in times[0] if not isinstance(times[0][k],list)];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(times)
 print('CACHE_PROGRESS',len(good),'successful jobs;',len(times),'complete A/B shapes;',len(profiles),'profiles')
 for r in times:print('TIME',r['id'],round(r['before_ms'],6),round(r['noslc_ms'],6),f"{r['delta_pct']:+.2f}%")
 for r in profiles:print('PROFILE',r['id'],r['variant'],'HBM_MB',round(r['hbm_read_MB'],2),'L2%',round(r['l2_hit_pct'],2),'latency',round(r['tcp_tcc_read_latency'],1))

if __name__=='__main__':main()
