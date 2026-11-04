"""Fail closed on environment changes; recovery is a separate fresh invocation."""
import datetime,fcntl,json,os,re,resource,subprocess,time
from pathlib import Path
import run as r
MARK=r.ROOT/'environment_recovery_required.json'
def check(key,attempt):
 t=time.monotonic()
 try:
  p=subprocess.run(['hy-smi'],env=r.ENV,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
  out=re.sub(r'\x1b\[[0-9;]*m','',p.stdout);rc=p.returncode
 except Exception as ex:out=repr(ex);rc=-1
 rows=[s.split() for s in out.splitlines() if re.match(r'^\s*4\s',s)]
 pct=rows[0][6] if len(rows)==1 and len(rows[0])>6 else None
 mem=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])
 row=dict(at=r.stamp(),key=key,attempt=attempt,rc=rc,hcu4=pct,mem_available_kib=mem,seconds=time.monotonic()-t,output=out)
 r.append(r.ROOT/'gate.jsonl',row)
 if mem<64*1024*1024:raise RuntimeError('LOW_HOST_MEMORY; save/stop')
 healthy=rc==0 and 'MKFD' not in out and pct is not None
 if healthy and pct!='0.0%':raise RuntimeError('CARD4_BUSY; save/stop, never switch')
 return healthy,row
def pause(key):
 t=time.monotonic();time.sleep(60);r.append(r.ROOT/'waits.jsonl',dict(at=r.stamp(),key=key,reason='environment_recovery_observation',seconds=time.monotonic()-t))
def gate(key):
 if MARK.exists():raise RuntimeError('RECOVERY_REQUIRED; fresh health.py recover before resuming A/B with new keys')
 for a in range(1,4):
  ok,row=check(key,a)
  if ok:
   if a==1:return
   raise RuntimeError('ENVIRONMENT_RETURNED; separate recovery process/probe and complete A/B remeasure required')
  if not MARK.exists():MARK.write_text(json.dumps(dict(first_failure=row,at=r.stamp()),indent=2))
  if a<3:pause(key)
 raise RuntimeError('ENVIRONMENT_FAILED_3; save/stop, no service manipulation')
def recover():
 if not MARK.exists():raise RuntimeError('No recovery marker; do not manufacture a recovery epoch')
 key='recovery_'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')
 ok,one=check(key,1)
 if not ok:raise RuntimeError('Recovery first check failed')
 pause(key);ok,two=check(key,2)
 if not ok:raise RuntimeError('Recovery second check failed')
 folder=r.ROOT/'runs'/key;folder.mkdir()
 known=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines() if x]
 prior=next(x for x in known if x['key']=='acc_cache_id76_base' and x.get('ok'))
 assert r.sha(r.ROOT/'bench_cache')==prior['benchmark_sha256'],'Known-good probe binary changed'
 cmd=[str(r.ROOT/'bench_cache'),'--dtype=bf16','--layout=NN','--m=1','--n=2048','--k=768','--groups=4','--backend=both','--blas-lib='+r.BLAS,'--include-args-build=0','--time=0','--validate=1','--correctness-repeats=3']
 env=dict(r.ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='0',LD_LIBRARY_PATH=str(Path(r.BLAS).parent)+':'+r.ENV.get('LD_LIBRARY_PATH',''))
 with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 out=(folder/'run.log').read_text();passed=p.returncode==0 and 'SELECT instance_id=5000' in out and 'SUMMARY result=PASS' in out and bool(re.search(r'ACCURACY backend=ck_vs_blas .*?failed=0(?:\s|$)',out))
 new_epoch=r.current_epoch()+1
 recovery_finished_at=r.stamp()
 r.append(r.ROOT/'recovery.jsonl',dict(at=recovery_finished_at,environment_epoch=new_epoch,key=key,known_good_prior_key=prior['key'],command=cmd,path=str(folder),ok=passed,rc=p.returncode,marker=json.loads(MARK.read_text()),checks=[one,two],benchmark_sha256=r.sha(cmd[0])))
 if not passed:raise RuntimeError('Known-good probe failed; recovery incomplete')
 (r.ROOT/'environment_epoch.json').write_text(json.dumps(dict(id=new_epoch,recovery_key=key,started_after=recovery_finished_at),indent=2))
 MARK.rename(r.ROOT/(key+'_resolved.json'))
 print('RECOVERY_OBSERVATION_PASS; observations do not guarantee future stability; remeasure affected full A/B with fresh tag')
if __name__=='__main__':
 import sys
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 if sys.argv[1:]!=['recover']:raise SystemExit('usage: health.py recover')
 with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);recover()
