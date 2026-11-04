import fcntl,json,resource,subprocess,time
from pathlib import Path
import run as r
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
B=r.ROOT/'small_precheck_v2'
specs=[(64,1,64,128,1),(64,7,256,160,4),(64,65,256,768,4),(64,8,6144,2048,32),(64,1,64,128,1024),(128,1,128,64,4),(128,65,256,80,4),(128,128,256,512,64),(128,129,256,768,4),(128,256,256,512,4),(128,1,128,64,1024),(32,1,128,128,4),(32,65,256,768,4)]
with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for spec in specs:
  key='probe2_'+'_'.join(map(str,spec));r.gate(key)
  folder=r.ROOT/'runs'/key/'1';folder.mkdir(parents=True,exist_ok=False)
  cmd=[str(B),*map(str,spec)];start=time.monotonic()
  with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,env=r.ENV,stdout=f,stderr=subprocess.STDOUT,timeout=300)
  rec=dict(key=key,at=r.stamp(),kind='mapping_probe_v2',command=cmd,path=str(folder),rc=p.returncode,seconds=time.monotonic()-start,ok=p.returncode==0,binary_sha256=r.sha(B),script_sha256=r.sha(__file__))
  r.append(r.ROOT/'jobs.jsonl',rec);print(key,p.returncode,flush=True)
  if p.returncode:raise RuntimeError('STOP '+str(folder))
print('PROBE_V2_PASS',len(specs),flush=True)
