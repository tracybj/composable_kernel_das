#!/usr/bin/env python3
"""Incremental small-M campaign. Fixed card 4; no full-dataset GPU run."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BENCH=Path('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
BASE=Path('/workspace/v7_coverage/p6b/bench_base')
BLAS='/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so'
TUNING=Path('/workspace/Grouped_Geem/blaslt_tuning_tools')
ENV=dict(os.environ,HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',
 CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN='0',
 CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M='0')

def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def gate(key):
 for attempt in range(1,4):
  r=subprocess.run(['hy-smi'],env=ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
  out=re.sub(r'\x1b\[[0-9;]*m','',r.stdout)
  rows=[s.split() for s in out.splitlines() if re.match(r'^\s*4\s',s)]
  pct=rows[0][6] if len(rows)==1 and len(rows[0])>6 else None
  append(ROOT/'gate.jsonl',dict(key=key,attempt=attempt,rc=r.returncode,hcu4=pct,output=out))
  if r.returncode==0 and 'MKFD' not in out and pct is not None:
   if pct!='0.0%':raise RuntimeError('Card4 busy: '+pct)
   return
  print('GATE_RETRY',key,attempt,flush=True)
  if attempt<3:time.sleep(60)
 raise RuntimeError('Gate failed three times; stop')

def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['accuracy','newblas','perf']);args=ap.parse_args()
 plan=json.loads((ROOT/'plan.json').read_text())
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for path in [BENCH,BASE,Path(BLAS),TUNING/'build/libgroupgemm_tune.so']:
  assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path)]['sha256'],path
 cases={int(r['ID']):r for r in csv.DictReader(Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv').open(encoding='utf-8-sig'))}
 jobs=[]
 def bench(i,rnd,tile=0,accuracy=False):
  c=cases[i];tag=f'accuracy_id{i}_n{tile}' if accuracy else f'perf_r{rnd}_id{i}_n{tile}'
  binary=BASE if tile==0 else BENCH
  cmd=[str(binary),'--dtype=bf16','--layout=NN',*[f'--{arg}={c[k]}' for arg,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],
   '--include-args-build=0',f'--backend={"both" if accuracy else "ck"}',
   '--blas-lib='+BLAS,f'--time={0 if accuracy else 1}','--validate=1',
   f'--correctness-repeats={3 if accuracy else 1}','--warmup=10','--samples=30']
  jobs.append(dict(key=tag,id=i,round=rnd,tile=tile,command=cmd,kind='accuracy' if accuracy else 'perf',expected=5000 if tile==0 else 5003 if tile==128 else 5004))
 if args.mode=='accuracy':
  for i in range(1179,1188):
   for tile in [128,64]:bench(i,0,tile,True)
 if args.mode=='perf':
  for rnd in [1,2,3]:
   for i in plan['new_ids']+plan['old_optimized_ids']:
    bench(i,rnd)
    if i in plan['old_optimized_ids'] or i<=1187:bench(i,rnd,128)
    if 1179<=i<=1187:
     # Independent before/after pair for exploratory N64 on new shapes.
     base=jobs[-2].copy();base['key']+= '_for64';jobs.append(base)
     bench(i,rnd,64)
 if args.mode=='newblas':
  for i in plan['new_ids']:
   jobs.append(dict(key=f'newblas_id{i}',id=i,kind='newblas',command=[sys.executable,'-u',str(TUNING/'tuning_common.py'),'--worker',str(ROOT/'new_blas/run_config.json'),str(i)]))
 with (Path('/workspace/v7_coverage/p6b')/'run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  journal=ROOT/'jobs.jsonl';records=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
  done={r['key'] for r in records if r.get('ok')}
  for number,spec in enumerate(jobs,1):
   if spec['key'] in done:continue
   gate(spec['key'])
   parent=ROOT/'runs'/spec['key'];parent.mkdir(parents=True,exist_ok=True)
   folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir()
   env=dict(ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M=str(spec.get('tile',0)),
    LD_LIBRARY_PATH=str(Path(BLAS).parent)+':'+ENV.get('LD_LIBRARY_PATH',''))
   if spec['kind']=='newblas':env['LD_PRELOAD']=str(Path(BLAS).resolve())
   record=dict(spec,path=str(folder),at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
   print('START',number,'/',len(jobs),spec['key'],flush=True);start=time.monotonic()
   try:
    with (folder/'run.log').open('w') as f:proc=subprocess.run(spec['command'],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=1800)
    out=(folder/'run.log').read_text(errors='replace')
    record.update(rc=proc.returncode,seconds=time.monotonic()-start,ok=proc.returncode==0)
    if spec['kind']=='newblas':
     result=ROOT/'new_blas/cases'/f"{spec['id']:04d}"/'result.json'
     record['result']=json.loads(result.read_text()) if result.exists() else None
     record['ok'] &= bool(record['result']) and record['result']['status']=='PASS'
    else:
     selected=re.search(r'SELECT instance_id=(\d+)',out);record['selected']=int(selected[1]) if selected else None
     record['ok'] &= record['selected']==spec['expected'] and 'SUMMARY result=PASS' in out
     if spec['kind']=='accuracy':
      matches=re.findall(r'ACCURACY backend=ck_vs_blas repeat=\d+ checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)',out)
      record['comparisons']=matches;record['ok'] &= len(matches)==3 and all(int(f)==0 and float(a)==0 for n,f,a in matches)
     else:
      m=re.search(r'PERF backend=ck_device_args (.*)',out)
      record['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
      record['ok'] &= bool(m) and bool(re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',out))
   except Exception as exc:
    record.update(ok=False,error=repr(exc),seconds=time.monotonic()-start)
   append(journal,record)
   print('DONE',spec['key'],'ok='+str(record['ok']),'sec='+str(round(record['seconds'],2)),flush=True)
   if not record['ok']:raise RuntimeError('STOP inspect '+str(folder/'run.log'))
 print('ALL_DONE',args.mode,flush=True)

if __name__=='__main__':main()
