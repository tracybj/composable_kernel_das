#!/usr/bin/env python3
"""Eight-hour session: fresh-process card4 gate, serial jobs, durable resume."""
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
REPO=Path('/workspace/composable_kernel')
BENCH=REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
BLAS='/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so'
ENV=dict(os.environ,HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN='0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M='0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_A_NOSLC='0')
GC='speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state'

def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gate(key):
 from health import gate as checked_gate
 return checked_gate(key)

def current_epoch():
 from journalview import epoch
 return epoch()

def cases():
 return {int(r['ID']):dict(id=int(r['ID']),**{k:int(r[k]) for k in ['M','N','K','G']}) for r in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig'))}

def main():
 global BENCH
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['cache_accuracy','cache_perf','cache_profile','cache_expand','cache_walk','regression']);ap.add_argument('--binary');ap.add_argument('--tag',default='');ap.add_argument('--repair-epoch');args=ap.parse_args()
 if args.binary:BENCH=Path(args.binary)
 snapshot=ROOT/'scripts'/f'{sha(__file__)}.py';snapshot.parent.mkdir(exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
 session=json.loads((ROOT/'session.json').read_text());deadline=datetime.datetime.fromisoformat(session['deadline_utc']).timestamp()
 cs=cases();jobs=[]
 def bench(key,c,variant='base',kind='accuracy',extra=None):
  timing=kind in ['perf','profile'];profile=kind=='profile'
  cmd=[str(BENCH),'--dtype=bf16','--layout=NN',*[f'--{a}={c[k]}' for a,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--include-args-build=0',f'--backend={"ck" if timing else "both"}',f'--blas-lib={BLAS}',f'--time={int(timing)}',f'--validate={int(not profile)}',f'--correctness-repeats={1 if timing or c["id"]==1178 else 3}',f'--warmup={1 if profile else 10}',f'--samples={1 if profile else 30}']
  if extra:cmd+=extra
  expected=5005 if variant=='noslc_row' else (5006 if c['K']>4096 else 5005) if variant=='noslc' else (5002 if c['K']>4096 else 5000)
  jobs.append(dict(key=key,id=c['id'],shape=c,variant=variant,kind=kind,command=cmd,expected=expected))
 if args.suite=='cache_accuracy':
  for i in [76,358,962,1126,1178]:
   for variant in ['base','noslc']:bench(f'acc_cache_id{i}_{variant}',cs[i],variant)
  for tag,c in [('m_tail',dict(id=-1,M=257,N=512,K=80,G=4)),('k64',dict(id=-2,M=1,N=256,K=64,G=4)),('g1024',dict(id=-3,M=1,N=256,K=64,G=1024))]:
   bench('acc_cache_'+tag,c,'noslc')
 if args.suite=='cache_perf':
  for rnd in [1,2,3]:
   for i in [1178,1177,1126,1164,962,966,1118,1095,358,76]:
    for variant in ['base','noslc']:bench(f'perf_cache_r{rnd}_id{i}_{variant}',cs[i],variant,'perf')
 if args.suite=='cache_profile':
  for i in [1178,1126,962]:
   for variant in ['base','noslc']:bench(f'profile_cache_id{i}_{variant}',cs[i],variant,'profile')
 if args.suite=='cache_expand':
  # Fixed before this campaign: all212 original K>4096 cases, all15 new,
  # and four short-K controls. Outside K>4096 is diagnostic, not the policy.
  ids=sorted({i for i,c in cs.items() if c['K']>4096}|set(range(1179,1194))|{76,358,962,966})
  (ROOT/'cache_expand_plan.json').write_text(json.dumps(dict(ids=ids,count=len(ids),strategy='Apply no-SLC only to common nonragged NN K>4096; new15 and other shortK controls are outside-policy diagnostics',benchmark_sha256=sha(BENCH)),indent=2))
  for rnd in [1,2,3]:
   for i in ids:
    for variant in ['base','noslc']:bench(f'expand_cache_r{rnd}_id{i}_{variant}',cs[i],variant,'perf')
 if args.suite=='cache_walk':
  for rnd in [1,2,3]:
   for i in [1178,1126,1164]:
    for variant in ['noslc','noslc_row']:bench(f'walk_cache_r{rnd}_id{i}_{variant}',cs[i],variant,'perf')
 if args.suite=='regression':
  for tag,binary in [('selector','tile_example_grouped_gemm_selector_test'),('query','tile_example_grouped_gemm_device_args_query_test'),('smoke','tile_example_grouped_gemm_device_args_smoke')]:
   jobs.append(dict(key='regression_'+tag,id=0,variant='base',kind=tag,command=[str(REPO/'build/bin'/binary)]))
 if args.tag and args.repair_epoch:raise RuntimeError('Use tag or repair-epoch, not both')
 if args.tag:
  for job in jobs:job['key']=args.tag+'_'+job['key']
 for job in jobs:
  job['logical_key']=job['key']
  if args.repair_epoch:job['key']=args.repair_epoch+'_'+job['key']
 with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from journalview import read as read_journal
  journal=ROOT/'jobs.jsonl';records,valid=read_journal()
  done=set(valid)
  if (ROOT/'invalidations.json').exists() and not args.repair_epoch and not args.tag and args.suite=='cache_expand':
   raise RuntimeError('Use --repair-epoch with fresh physical keys after invalidation')
  binary_hash=sha(BENCH)
  for number,spec in enumerate(jobs,1):
   if spec['logical_key'] in done:continue
   if time.time()>deadline-900:raise RuntimeError('DEADLINE_RESERVE; stop new GPU jobs')
   assert sha(BENCH)==binary_hash,'Benchmark changed during campaign'
   gate(spec['key'])
   parent=ROOT/'runs'/spec['key'];parent.mkdir(parents=True,exist_ok=True)
   folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir()
   env=dict(ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=str(int(spec['variant'] in ['noslc','noslc_row'])),CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8=str(int(spec['variant']=='noslc_row')),LD_LIBRARY_PATH=str(Path(BLAS).parent)+':'+ENV.get('LD_LIBRARY_PATH',''))
   # Query/smoke assert production defaults and manage their own opt-ins.
   # Do not leak the campaign's global V7 enable into those baseline checks.
   if spec['kind'] in ['query','smoke']:
    env['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN']='0'
   cmd=spec['command']
   if spec['kind']=='profile':cmd=['/opt/rocm-6.3.3/bin/xprof','--kernels','V7Uniform','--sections',GC,'--output-dir',str(folder/'profile')]+cmd
   record=dict(spec,environment_epoch=current_epoch(),at=stamp(),path=str(folder),command=cmd,benchmark_sha256=binary_hash,script_sha256=sha(__file__),environment={k:v for k,v in env.items() if k.startswith(('CK_TILE_GROUPED_GEMM_','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))})
   (folder/'command.json').write_text(json.dumps(record,indent=2))
   print('START',number,'/',len(jobs),spec['key'],flush=True);start=time.monotonic()
   try:
    with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=min(1800,max(30,int(deadline-time.time()))))
    out=(folder/'run.log').read_text(errors='replace');record.update(rc=p.returncode,seconds=time.monotonic()-start,ok=p.returncode==0)
    if spec['kind'] in ['accuracy','perf','profile']:
     sel=re.search(r'SELECT instance_id=(\d+)',out);record['selected']=int(sel[1]) if sel else None
     record['ok'] &= record['selected']==spec['expected'] and 'SUMMARY result=PASS' in out
     if spec['kind']=='accuracy':
      checks=re.findall(r'ACCURACY backend=ck_vs_blas repeat=(\d+) checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)',out)
      record['comparisons']=checks;record['ok'] &= bool(checks) and all(int(x[2])==0 for x in checks)
     elif spec['kind']=='perf':
      m=re.search(r'PERF backend=ck_device_args (.*)',out)
      record['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
      record['ok'] &= bool(m) and bool(re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',out))
     else:
      record['perf_files']=[str(p) for p in (folder/'profile').glob('*.perf')];record['ok'] &= len(record['perf_files'])==1
   except Exception as exc:record.update(ok=False,error=repr(exc),seconds=time.monotonic()-start)
   append(journal,record)
   print('DONE',spec['key'],'ok='+str(record['ok']),'sec='+str(round(record['seconds'],2)),flush=True)
   if not record['ok']:raise RuntimeError('STOP inspect '+str(folder/'run.log')+'; no automatic kernel-error retries')
 print('ALL_DONE',args.suite,flush=True)

if __name__=='__main__':main()
