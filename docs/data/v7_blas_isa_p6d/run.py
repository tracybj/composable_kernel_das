#!/usr/bin/env python3
"""Serial card-4 representative diagnostics; never a full-shape campaign."""
import argparse
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
XPROF='/opt/rocm-6.3.3/bin/xprof'
BLAS='/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so.0.10'
GC='speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state'
ENV=dict(os.environ,HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN='0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M='0')

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
   if pct!='0.0%':raise RuntimeError('Card4 busy: '+pct+'; stop without switching card')
   return
  print('GATE_RETRY',key,attempt,flush=True)
  if attempt<3:time.sleep(60)
 raise RuntimeError('Gate failed three times')

def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['timing','profile','trace']);args=ap.parse_args()
 cases=json.loads((ROOT/'cases.json').read_text());jobs=[]
 for path,v in json.loads((ROOT/'manifest.json').read_text()).items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==v['sha256'],path
 rounds=[1,2,3] if args.mode=='timing' else [1]
 for rnd in rounds:
  for c in cases:
   i=c['id']
   if args.mode=='profile' and i not in [146,772,1186,1187,1178]:continue
   if args.mode=='trace' and i!=681:continue
   for backend in ['base','small128','blas']:
    if backend=='small128' and i in [146,1178]:continue
    if args.mode=='trace' and backend!='small128':continue
    timing=args.mode=='timing';warmup,samples=(10,30) if timing else (1,1) if args.mode=='profile' else (0,1)
    if backend=='blas':
     cmd=[sys.executable,'-u',str(ROOT/'blas_replay.py'),f'--id={i}',f'--warmup={warmup}',f'--samples={samples}',f'--repeats={"1,20" if timing else "1"}',f'--validate={int(timing and rnd==1 and i!=1178)}']
     kernel=c['kernel_name'];expected=None
    else:
     cmd=[str(BENCH),'--dtype=bf16','--layout=NN',*[f'--{arg}={c[k]}' for arg,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--backend=ck','--time=1',f'--warmup={warmup}',f'--samples={samples}','--include-args-build=0',f'--validate={int(timing)}','--correctness-repeats=1']
     kernel='V7Uniform' if i!=146 else 'kentry'
     expected=5003 if backend=='small128' else 4003 if i==146 else 5002 if i==1178 else 5000
    key=f'{args.mode}_r{rnd}_id{i}_{backend}'
    jobs.append(dict(key=key,id=i,round=rnd,backend=backend,command=cmd,kernel=kernel,expected=expected))
 with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  journal=ROOT/'jobs.jsonl';records=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
  done={r['key'] for r in records if r.get('ok')}
  for number,spec in enumerate(jobs,1):
   if spec['key'] in done:continue
   gate(spec['key'])
   parent=ROOT/'runs'/spec['key'];parent.mkdir(parents=True,exist_ok=True)
   folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir()
   env=dict(ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='128' if spec['backend']=='small128' else '0',LD_LIBRARY_PATH=str(Path(BLAS).parent)+':'+ENV.get('LD_LIBRARY_PATH',''))
   if spec['backend']=='blas':env['LD_PRELOAD']=BLAS
   cmd=spec['command']
   if args.mode!='timing':
    options=['--sections',GC] if args.mode=='profile' else ['--enable-sqtt','--enable-raw','--target-se','0','--target-cu','0','--trace-size','4','--replay-mode','off']
    cmd=[XPROF,'--kernels',spec['kernel'],'--output-dir',str(folder/'profile'),*options]+cmd
   record=dict(spec,mode=args.mode,path=str(folder),command=cmd,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scripts_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run.py',ROOT/'blas_replay.py']})
   (folder/'command.json').write_text(json.dumps(record,indent=2))
   print('START',number,'/',len(jobs),spec['key'],flush=True);start=time.monotonic()
   try:
    with (folder/'run.log').open('w') as f:proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=1200)
    out=(folder/'run.log').read_text(errors='replace');record.update(rc=proc.returncode,seconds=time.monotonic()-start,ok=proc.returncode==0)
    if spec['backend']=='blas':
     lines=[s[7:] for s in out.splitlines() if s.startswith('RESULT ')]
     record['result']=json.loads(lines[-1]) if lines else None;record['ok'] &= bool(lines)
    else:
     selected=re.search(r'SELECT instance_id=(\d+)',out);record['selected']=int(selected[1]) if selected else None
     m=re.search(r'PERF backend=ck_device_args (.*)',out)
     record['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
     record['ok'] &= record['selected']==spec['expected'] and 'SUMMARY result=PASS' in out
     if args.mode=='timing':record['ok'] &= bool(m) and bool(re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',out))
    if args.mode!='timing':
     record['perf_files']=[str(p) for p in (folder/'profile').glob('*.perf')];record['ok'] &= bool(record['perf_files'])
   except Exception as exc:record.update(ok=False,error=repr(exc),seconds=time.monotonic()-start)
   append(journal,record)
   print('DONE',spec['key'],'ok='+str(record['ok']),'sec='+str(round(record['seconds'],2)),flush=True)
   if not record['ok']:raise RuntimeError('STOP inspect '+str(folder/'run.log'))
 print('ALL_DONE',args.mode,flush=True)

if __name__=='__main__':main()
