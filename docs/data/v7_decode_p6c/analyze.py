#!/usr/bin/env python3
"""Splice the predeclared experimental range into historical distributions."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st

ROOT=Path(__file__).resolve().parent
SHAPES=Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv')
BLAS=Path('/workspace/bf16NN-groupgemm.csv')
LABELS=['[0.99,inf)','[0.8,0.99)','[0.5,0.8)','[0.3,0.5)','[0.2,0.3)','<0.2']

def load_old(path):
 rows={}
 for line in Path(path).read_text().splitlines():
  f=line.split();m=re.search(r' ms=([0-9.eE+-]+)',line)
  if m and 'result=PASS rc=0' in line:rows[int(f[0])]=float(m[1])
 assert len(rows)==1178
 return rows

def bins(ratios):
 result=[sum(r>=.99 for r in ratios),sum(.8<=r<.99 for r in ratios),sum(.5<=r<.8 for r in ratios),sum(.3<=r<.5 for r in ratios),sum(.2<=r<.3 for r in ratios),sum(r<.2 for r in ratios)]
 assert sum(result)==len(ratios)
 return result

def main():
 plan=json.loads((ROOT/'plan.json').read_text())
 jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
 good={r['key']:r for r in jobs if r.get('ok')}
 assert len(good)==3759,(len(good),len(jobs))
 failed_attempts=[r for r in jobs if not r['ok']]
 for r in failed_attempts:
  log=(Path(r['path'])/'run.log').read_text()
  assert 'invalid device ordinal' in log and r.get('selected') is None and r.get('result') is None,r['key']
  assert 'PERF backend=' not in log and 'ACCURACY backend=' not in log,r['key']
 kinds=collections.Counter(r['kind'] for r in good.values())
 assert kinds==dict(accuracy=18,newblas=15,perf=3726),kinds
 shapes={int(r['ID']):r for r in csv.DictReader(SHAPES.open(encoding='utf-8-sig'))}
 blas={int(r['ID']):float(r['BLAS ms']) for r in csv.DictReader(BLAS.open(encoding='utf-8-sig'))}
 old10=load_old('/workspace/v7_coverage/full1178/perf/summary.txt')
 old11=load_old('/workspace/v7_coverage/full1178_p4b/perf/summary.txt')
 native=[]
 for i in plan['new_ids']:
  r=good[f'newblas_id{i}']['result'];assert r['status']=='PASS' and r['validation']['status']=='PASS'
  blas[i]=r['time_ms'];native.append(r)
 before=dict(old11);after=dict(old11);details=[]
 for i in plan['old_optimized_ids']+plan['new_ids']:
  b=[good[f'perf_r{rnd}_id{i}_n0']['result'] for rnd in [1,2,3]]
  before[i]=st.median(r['median_ms'] for r in b);after[i]=before[i]
  c=shapes[i];policy=i in plan['old_optimized_ids'] or i in plan['optimized_new_ids']
  for tile in [128,64]:
   keys=[f'perf_r{rnd}_id{i}_n{tile}' for rnd in [1,2,3]]
   if not all(k in good for k in keys):continue
   a=[good[k]['result'] for k in keys];a_ms=st.median(r['median_ms'] for r in a)
   ref=b if tile==128 else [good[f'perf_r{rnd}_id{i}_n0_for64']['result'] for rnd in [1,2,3]]
   b_ms=st.median(r['median_ms'] for r in ref)
   if policy and tile==128:after[i]=a_ms
   details.append(dict(ID=i,**{k:int(c[k]) for k in ['M','N','K','G']},tile_n=tile,apply_policy=policy and tile==128,
    before_ms=b_ms,after_ms=a_ms,delta_pct=100*(a_ms/b_ms-1),speedup=b_ms/a_ms,blas_ms=blas[i],
    R_before=blas[i]/b_ms,R_after=blas[i]/a_ms,before_round_ms=[r['median_ms'] for r in ref],after_round_ms=[r['median_ms'] for r in a],
    before_cv=[r['cv'] for r in ref],after_cv=[r['cv'] for r in a]))
 assert len(details)==618 and len(before)==len(after)==1193
 assert sum(r['apply_policy'] for r in details)==607
 datasets=[('10-30',old10),('11-03',old11),('current_control_original1178',{i:t for i,t in before.items() if i<=1178}),
  ('current_optimized_original1178',{i:t for i,t in after.items() if i<=1178}),('current_control_extended1193',before),('current_optimized_extended1193',after),
  ('new15_control',{i:before[i] for i in plan['new_ids']}),('new15_optimized',{i:after[i] for i in plan['new_ids']})]
 tables=[]
 for name,times in datasets:
  for phase in ['prefill','decode']:
   rr=[blas[i]/ms for i,ms in times.items() if shapes[i]['phase']==phase]
   tables.append(dict(dataset=name,phase=phase,total_case=len(rr),buckets=bins(rr),median_R=st.median(rr),R_lt_08=sum(r<.8 for r in rr),R_lt_05=sum(r<.5 for r in rr)))
 expected11={'prefill':[244,54,8,0,0,0],'decode':[308,415,113,26,8,2]}
 assert all(r['buckets']==expected11[r['phase']] for r in tables if r['dataset']=='11-03')
 old_route={}
 for line in Path('/workspace/v7_coverage/full1178_p4b/perf/summary.txt').read_text().splitlines():
  match=re.search(r' inst=(\d+) ',line)
  if match and 'result=PASS rc=0' in line:old_route[int(line.split()[0])]=int(match[1])
 assert len(old_route)==1178
 merged=[]
 for i in range(1,1194):
  c=shapes[i];merged.append(dict(ID=i,**{k:c[k] for k in ['M','N','K','G','phase']},CK_before_ms=before[i],CK_after_ms=after[i],BLAS_ms=blas[i],R_before=blas[i]/before[i],R_after=blas[i]/after[i],
    CK_source='current_AB' if i in plan['old_optimized_ids'] or 1179<=i<=1187 else 'current_new_baseline' if i>=1188 else '11-03_carried',BLAS_gpu=0 if i<=1178 else 4,
    instance_after=5003 if i in plan['old_optimized_ids'] or i in plan['optimized_new_ids'] else 5000 if i>=1179 else old_route[i]))
 def save_csv(path,rows):
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 save_csv(ROOT/'merged1193.csv',merged)
 fields=['ID','M','N','K','G','tile_n','apply_policy','before_ms','after_ms','delta_pct','speedup','blas_ms','R_before','R_after']
 save_csv(ROOT/'measured.csv',[{k:r[k] for k in fields} for r in details])
 for name,data in [('tables.json',tables),('measured.json',details),('native_new15.json',native)]:
  (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 gates=[json.loads(s) for s in (ROOT/'gate.jsonl').read_text().splitlines()]
 assert all(any(g['key']==k and g['rc']==0 and g['hcu4']=='0.0%' for g in gates) for k in good)
 comparisons=[row for r in good.values() if r['kind']=='accuracy' for row in r['comparisons']]
 applied=[r for r in details if r['apply_policy']]
 checks=dict(valid_jobs=len(good),counts=dict(kinds),unique_CK_shapes=615,old_shape_remeasured=600,old_shape_carried=578,new_shapes=15,
  applied_shapes=607,full_output_comparisons=len(comparisons),full_output_elements=sum(int(n) for n,failed,err in comparisons),
  failed=0,max_abs=0,environment_failed_attempts=len(failed_attempts),total_attempts=len(jobs),gate_records=len(gates),median_delta_pct=st.median(r['delta_pct'] for r in applied),regressions_gt2=sum(r['delta_pct']>2 for r in applied),
  regressions_gt5=sum(r['delta_pct']>5 for r in applied),improved_gt2=sum(r['delta_pct']< -2 for r in applied),
  one_call_before_ms=sum(r['before_ms'] for r in applied),one_call_after_ms=sum(r['after_ms'] for r in applied),
  one_call_delta_pct=100*(sum(r['after_ms'] for r in applied)/sum(r['before_ms'] for r in applied)-1),
  geomean_delta_pct=100*(math.exp(st.mean(math.log(r['after_ms']/r['before_ms']) for r in applied))-1))
 (ROOT/'validation.json').write_text(json.dumps(checks,indent=2))
 print('INCREMENTAL_CAMPAIGN_PASS',json.dumps(checks))
 for r in tables:print(r['dataset'],r['phase'],r['total_case'],r['buckets'],'medianR',round(r['median_R'],4))
 print('NEW_SHAPES')
 for r in details:
  if r['ID']>=1179:print(r['ID'],r['M'],r['tile_n'],round(r['delta_pct'],2),round(r['R_after'],4))

if __name__=='__main__':main()
