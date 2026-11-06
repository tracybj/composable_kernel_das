"""Audit accepted complete contrasts and summarize all measured outcomes, never fastest arms."""
import csv,json,statistics,re,datetime,math
from pathlib import Path
import runtime as r
import campaign as c
root=r.ROOT;binary=r.sha(c.BENCH)
jobs=[json.loads(x) for x in (root/'jobs.jsonl').read_text().splitlines()] if (root/'jobs.jsonl').exists() else []
keys={x['key']:x for x in jobs};assert len(keys)==len(jobs)
rows=[json.loads(x) for x in (root/'shape_results.jsonl').read_text().splitlines()] if (root/'shape_results.jsonl').exists() else []
rows=[x for x in rows if x['binary_sha256']==binary]
complete_keys={k for row in rows for k in row['keys']}
partial_current=[x['key'] for x in jobs if x['binary_sha256']==binary and x.get('kind') in ['accuracy','perf'] and x['key'] not in complete_keys]
plans=[]
for path in sorted(root.glob('plan_*_'+binary[:12]+'.json')):
 spec=json.loads(path.read_text());done={x['id'] for x in rows if x['suite']==spec['suite']}
 plans.append(dict(suite=spec['suite'],planned=len(spec['cases']),complete=len(done),missing=[x['id'] for x in spec['cases'] if x['id'] not in done]))
results=[]
for row in rows:
 group=[keys[k] for k in row['keys']]
 assert all(x['ok'] and x['physical_gpu']==4 and x['epoch']==row['epoch'] and x['binary_sha256']==binary and not x['kernel_service_events'] for x in group)
 for job in group:
  assert job['command'][0]==str(c.BENCH)
  assert job['environment']['HIP_VISIBLE_DEVICES']==job['environment']['CUDA_VISIBLE_DEVICES']=='4'
  gate=next(x for x in map(json.loads,(root/'gate.jsonl').read_text().splitlines()) if x['key']==job['key'])
  assert gate['rc']==0 and gate['hcu']=='0.0%' and gate['mem_available_kib']>=64*1024*1024 and 'MKFD' not in gate['output'].upper()
  text=(Path(job['path'])/'run.log').read_text();assert 'MKFD' not in text.upper()
  assert '--include-args-build=0' in job['command'] and '--launches-per-sample=1' in job['command']
  assert job['selected']==job['expected']
  if job['kind']=='accuracy':
   elements=math.prod(job['shape'][k] for k in ['M','N','G'])
   assert all(int(x[1])==elements and int(x[2])==0 for x in job['comparisons'])
   for backend in ['ck_redzone','blas_redzone']:
    checks=re.findall('ACCURACY backend='+backend+r' repeat=\d+ checked=(\d+) failed=(\d+)',text)
    assert len(checks)==len(job['comparisons']) and all(int(n)==32 and int(f)==0 for n,f in checks)
 if row.get('accuracy_only'):continue
 assert len(group)==8 and sum(x['kind']=='accuracy' for x in group)==2
 accuracy={x['variant']:x for x in group if x['kind']=='accuracy'}
 first_perf=min(datetime.datetime.fromisoformat(x['at']) for x in group if x['kind']=='perf')
 assert all(datetime.datetime.fromisoformat(x['finished_at'])<first_perf for x in accuracy.values())
 for variant in ['base','new']:
  perf=[x for x in group if x['kind']=='perf' and x['variant']==variant]
  assert len(perf)==3 and [x['result']['median_ms'] for x in perf]==row['times'][variant]
  assert statistics.median(row['times'][variant])==row[variant+'_ms']
 paired=[(new/base-1)*100 for base,new in zip(row['times']['base'],row['times']['new'])]
 high_cv=max(v for values in row['cvs'].values() for v in values)>.05
 result=dict(suite=row['suite'],ID=row['id'],**row['shape'],active=row['active'],base_instance=accuracy['base']['selected'],new_instance=accuracy['new']['selected'],base_ms=row['base_ms'],new_ms=row['new_ms'],change_pct=row['change_pct'],round_changes_pct=paired,three_round_improvement_gt2=all(x < -2 for x in paired),three_round_regression_gt2=all(x > 2 for x in paired),high_cv=high_cv,frozen_BLAS_ms=row['frozen_BLAS_ms'],R_new=row['R_new'],BLAS_source=row['BLAS_source'],epoch=row['epoch'],physical_gpu=4,binary_sha256=binary,round_ms=row['times'],round_cv=row['cvs'],keys=row['keys'],logs=[x['path']+'/run.log' for x in group])
 results.append(result)
summary={}
for suite in sorted({x['suite'] for x in results}):
 group=[x for x in results if x['suite']==suite];active=[x for x in group if x['active']]
 summary[suite]=dict(completed=len(group),active=len(active),fallback=len(group)-len(active),improved_gt2=sum(x['change_pct'] < -2 for x in active),regressed_gt2=sum(x['change_pct']>2 for x in active),within2=sum(abs(x['change_pct'])<=2 for x in active),three_round_improved_gt2=sum(x['three_round_improvement_gt2'] for x in active),three_round_regressed_gt2=sum(x['three_round_regression_gt2'] for x in active),high_cv=sum(x['high_cv'] for x in active),median_change_pct=statistics.median(x['change_pct'] for x in active) if active else None,sum_base_ms=sum(x['base_ms'] for x in active),sum_new_ms=sum(x['new_ms'] for x in active))
latest={}
for row in results:latest[row['ID']]=row # chronological complete series, no fastest selection
(root/'results.json').write_text(json.dumps(results,indent=2))
if results:
 with (root/'results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader()
  for x in results:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()})
(root/'latest_complete_by_shape.json').write_text(json.dumps(list(latest.values()),indent=2))
(root/'summary.json').write_text(json.dumps(dict(binary_sha256=binary,suites=summary,plans=plans,partial_current_binary_jobs=partial_current,prior_binary_jobs=sum(x['binary_sha256']!=binary for x in jobs),unique_measured_shapes=len(latest),all_raw_jobs=len(jobs),failed_jobs=[x['key'] for x in jobs if not x.get('ok')],note='Subset architecture experiment, not1193replay; historical Native baselines keep original GPU/repeat20 protocol'),indent=2))
print('DECODE_REPORT_AUDIT_PASS',json.dumps(summary),flush=True)
