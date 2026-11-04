#!/usr/bin/env python3
"""Fresh card4 reference, exact P6d frozen algorithm replay; no retuning."""
import argparse,datetime,fcntl,json,os,re,resource,subprocess,sys,time
from pathlib import Path
import run as r
import health
R=r.ROOT
ap=argparse.ArgumentParser();ap.add_argument('--repair-epoch');args=ap.parse_args()
deadline=datetime.datetime.fromisoformat(json.loads((R/'session.json').read_text())['deadline_utc']).timestamp()
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
script=R.parent/'p6d/blas_replay.py'
jobs=[]
for rnd in [1,2,3]:
 for i in [1178,76,633,681,358,772,1186,1187]:
  if i==1178:
   c=r.cases()[i];cmd=[str(r.BENCH),'--dtype=bf16','--layout=NN',*[f'--{x}={c[y]}' for x,y in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--backend=ck','--time=1','--warmup=10','--samples=30','--include-args-build=0','--validate=1','--correctness-repeats=1']
   jobs.append(dict(key=f'nativeref_r{rnd}_id{i}_noslc',id=i,round=rnd,variant='noslc',kind='native_reference_ck',command=cmd))
  cmd=[sys.executable,'-u',str(script),f'--id={i}','--warmup=10','--samples=30','--repeats=1,20',f'--validate={int(rnd==1 and i!=1178)}']
  jobs.append(dict(key=f'nativeref_r{rnd}_id{i}_blas',id=i,round=rnd,variant='blas',kind='native_reference',command=cmd))
for j in jobs:
 j['logical_key']=j['key']
 if args.repair_epoch:j['key']=args.repair_epoch+'_'+j['key']
with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 from journalview import read as read_journal
 prior,valid=read_journal();done=set(valid);bh=r.sha(r.BENCH)
 for num,j in enumerate(jobs,1):
  if j['logical_key'] in done:continue
  if time.time()>deadline-900:raise RuntimeError('Deadline reserve; no new Native jobs')
  assert r.sha(r.BENCH)==bh
  health.gate(j['key']);parent=R/'runs'/j['key'];parent.mkdir(parents=True,exist_ok=True);folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir()
  env=dict(r.ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC='1' if j['variant']=='noslc' else '0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='0',LD_LIBRARY_PATH=str(Path(r.BLAS).parent)+':'+r.ENV.get('LD_LIBRARY_PATH',''))
  if j['variant']=='blas':env['LD_PRELOAD']=r.BLAS
  rec=dict(j,environment_epoch=r.current_epoch(),at=r.stamp(),path=str(folder),benchmark_sha256=bh,script_sha256=r.sha(__file__),replay_sha256=r.sha(script),environment={k:v for k,v in env.items() if k.startswith(('CK_TILE_GROUPED_GEMM_','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','LD_PRELOAD'))})
  (folder/'command.json').write_text(json.dumps(rec,indent=2));print('START',num,'/',len(jobs),j['key'],flush=True);t=time.monotonic()
  try:
   with (folder/'run.log').open('w') as f:p=subprocess.run(j['command'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(900,max(30,int(deadline-time.time()))))
   out=(folder/'run.log').read_text(errors='replace');rec.update(rc=p.returncode,seconds=time.monotonic()-t,ok=p.returncode==0)
   if j['variant']=='blas':
    lines=[s[7:] for s in out.splitlines() if s.startswith('RESULT ')];rec['result']=json.loads(lines[-1]) if lines else None;rec['ok'] &= bool(lines)
   else:
    m=re.search(r'PERF backend=ck_device_args (.*)',out);rec['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
    rec['ok'] &= bool(m) and 'SELECT instance_id=5006' in out and 'SUMMARY result=PASS' in out
  except Exception as ex:rec.update(ok=False,error=repr(ex),seconds=time.monotonic()-t)
  r.append(R/'jobs.jsonl',rec);print('DONE',j['key'],rec['ok'],flush=True)
  if not rec['ok']:raise RuntimeError('STOP '+str(folder))
print('NATIVE_REFERENCE_PASS',len(jobs))
