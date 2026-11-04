#!/usr/bin/env python3
"""Default-selector experiments, separate baselines for causal comparisons."""
import argparse,datetime,fcntl,json,os,re,resource,subprocess,time
from pathlib import Path
import run as r
import health
ROOT=r.ROOT
IDS={'base':5000,'old128':5003,'old64':5004,'drain':5007,'k32':5008,'square':5009,'k32n64':5010}
EXPERIMENT={'drain':1,'k32':2,'square':3,'k32n64':4}
def main():
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 ap=argparse.ArgumentParser();ap.add_argument('suite',choices=['accuracy','stress','perf','profile','regression','expand']);ap.add_argument('--tag',default='');ap.add_argument('--plan');ap.add_argument('--repair-epoch');a=ap.parse_args()
 cs=r.cases();jobs=[]
 def add(key,c,v,kind='accuracy',extra=None,expected=None,contrast=None,round=None,repeats=None):
  timing=kind in ['perf','profile'];profile=kind=='profile'
  cmd=[str(r.BENCH),'--dtype=bf16','--layout=NN',*[f'--{x}={c[y]}' for x,y in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--include-args-build=0',f'--backend={"ck" if timing else "both"}','--blas-lib='+r.BLAS,f'--time={int(timing)}',f'--validate={int(not profile)}',f'--correctness-repeats={repeats or (1 if timing else 3)}',f'--warmup={1 if profile else 10}',f'--samples={1 if profile else 30}']
  if extra:cmd+=extra
  jobs.append(dict(key=key,shape=c,id=c['id'],variant=v,kind=kind,command=cmd,expected=IDS[v] if expected is None else expected,contrast=contrast,round=round))
 def shape(m,n,k,g):return dict(id=-1,M=m,N=n,K=k,G=g)
 if a.suite=='accuracy':
  for v in EXPERIMENT:
   for m in [1,7,63,64,65,127,128,129,255,256]:add(f'smallacc_{v}_m{m}',shape(m,256,512,4),v)
   for k in ([128,160,192,224,256,288,320,352,384,416,448,480,768,2048,4096] if v in ['k32','k32n64'] else [64,80,96,112,128,144,160,176,192,208,224,240,768,2048,4096]):add(f'smallacc_{v}_k{k}',shape(7,256,k,4),v)
   add(f'smallacc_{v}_g1024',shape(1,128,128 if v in ['k32','k32n64'] else 64,1024),v)
   add(f'smallacc_{v}_g32',cs[358],v)
   add(f'smallacc_{v}_overcover',shape(1,256,128 if v in ['k32','k32n64'] else 64,4),v,extra=['--num-cu=80'])
   for i in [1186,1187]:add(f'smallacc_{v}_newid{i}',cs[i],v)
   for tag,c,extra,expected in [
     ('n1232',shape(1,1232,1536,4),None,4003),('prefill',shape(512,256,512,4),None,5000),('largek',shape(1,256,4112,4),None,5002),
     ('ragged',shape(1,256,512,4),['--group-lengths=1,65,7,129'],5000),('empty',shape(1,256,512,4),['--group-lengths=1,65,0,7','--allow-empty-groups=1'],0),
     ('disable',shape(1,256,512,4),None,5000),('disable_all_small',shape(1,256,512,4),None,5000),('global_off',shape(1,256,512,4),None,0)]:
    add(f'smallacc_{v}_{tag}',c,v,extra=extra,expected=expected)
    if tag in ['disable','disable_all_small','global_off']:jobs[-1]['disable']=tag
   if v in ['k32','k32n64']:
    for k in [64,80,96,144]:add(f'smallacc_{v}_fallbackk{k}',shape(1,256,k,4),v,expected=5000)
   if v=='k32n64':add('smallacc_k32n64_n64',shape(1,64,128,4),v)
 if a.suite=='stress':
  for v in EXPERIMENT:
   add('smallstress_'+v+'_min',shape(1,128,128 if v in ['k32','k32n64'] else 64,4),v,repeats=20)
   add('smallstress_'+v+'_tail',shape(65,256,416 if v in ['k32','k32n64'] else 208,32),v,repeats=20)
 if a.suite=='perf':
  # Fixed representative plan before measurements; all 15 new shapes included.
  ids=list(dict.fromkeys([76,633,681,358,772,84,689,639,33,1,942,556]+list(range(1179,1194))))
  (ROOT/'small_perf_plan.json').write_text(json.dumps(dict(ids=ids,contrasts={'drain':'old128','k32':'old128','square':'old128','k32n64':'old64'}),indent=2))
  for rnd in [1,2,3]:
   for i in ids:
    c=cs[i]
    # Original route and old128 establish honest before/after table.
    add(f'smallperf_r{rnd}_id{i}_original',c,'base','perf',expected=5002 if c['K']>4096 else 5000,round=rnd,contrast='original')
    if c['M']>256:
     # Experimental switches must preserve prefill route, once per new prefill shape.
     add(f'smallperf_r{rnd}_id{i}_prefill',c,'k32','perf',expected=5000,round=rnd,contrast='prefill');continue
    for v,b in [('drain','old128'),('k32','old128'),('square','old128'),('k32n64','old64')]:
     if c['K']>4096 or (v in ['k32','k32n64'] and (c['K']<128 or c['K']%32)):continue
     for tag,which in [('before',b),('after',v)]:add(f'smallperf_r{rnd}_id{i}_{v}_{tag}',c,which,'perf',contrast=v,round=rnd)
 if a.suite=='profile':
  for i,vs in [(76,['old128','drain','k32','old64','k32n64']),(633,['old64','k32n64']),(681,['old128','k32']),(358,['old128','k32','k32n64']),(772,['old128','square']),(1187,['old128','square','k32'])]:
   for v in vs:add(f'smallprofile_id{i}_{v}',cs[i],v,'profile')
 if a.suite=='expand':
  if not a.plan:raise RuntimeError('Expansion requires predeclared attribute strategy plan')
  plan=json.loads(Path(a.plan).read_text());v=plan['variant'];ids=plan['ids'];baseline=plan.get('baseline','old128')
  for rnd in [1,2,3]:
   for i in ids:
    if plan.get('include_original'):
     add(f'smallexpand_{plan["name"]}_r{rnd}_id{i}_original',cs[i],'base','perf',contrast=plan['name'],round=rnd)
    for tag,which in [('before',baseline),('after',v)]:add(f'smallexpand_{plan["name"]}_r{rnd}_id{i}_{tag}',cs[i],which,'perf',contrast=plan['name'],round=rnd)
 if a.suite=='regression':
  for tag in ['selector_test','device_args_query_test','device_args_smoke']:
   jobs.append(dict(key='smallreg_'+tag,id=0,variant='base',kind='regression',command=[str(r.REPO/'build/bin'/('tile_example_grouped_gemm_'+tag))]))
 if a.tag and a.repair_epoch:raise RuntimeError('Use tag or repair-epoch, not both')
 if a.tag:
  for j in jobs:j['key']=a.tag+'_'+j['key']
 for j in jobs:
  j['logical_key']=j['key']
  if a.repair_epoch:j['key']=a.repair_epoch+'_'+j['key']
 snap=ROOT/'scripts'/(r.sha(__file__)+'.py');snap.write_bytes(Path(__file__).read_bytes())
 deadline=datetime.datetime.fromisoformat(json.loads((ROOT/'session.json').read_text())['deadline_utc']).timestamp()
 with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  from journalview import read as read_journal
  journal=ROOT/'jobs.jsonl';prior,valid=read_journal();done=set(valid);bh=r.sha(r.BENCH)
  for num,j in enumerate(jobs,1):
   if j['logical_key'] in done:continue
   if time.time()>deadline-900:raise RuntimeError('Deadline reserve')
   assert r.sha(r.BENCH)==bh,'Benchmark changed'
   health.gate(j['key'])
   v=j['variant'];env=dict(r.ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC='0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M=str(128 if v=='old128' else 64 if v=='old64' else 0),CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=str(EXPERIMENT.get(v,0)),CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='0',LD_LIBRARY_PATH=str(Path(r.BLAS).parent)+':'+r.ENV.get('LD_LIBRARY_PATH',''))
   if j.get('disable')=='disable':env['CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT']='1'
   if j.get('disable')=='disable_all_small':env['CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M']='1'
   if j.get('disable')=='global_off' or j['kind']=='regression':env['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN']='0'
   parent=ROOT/'runs'/j['key'];parent.mkdir(parents=True,exist_ok=True);folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir();cmd=j['command']
   if j['kind']=='profile':cmd=['/opt/rocm-6.3.3/bin/xprof','--kernels','V7Uniform','--sections',r.GC,'--output-dir',str(folder/'profile')]+cmd
   rec=dict(j,environment_epoch=r.current_epoch(),at=r.stamp(),command=cmd,path=str(folder),benchmark_sha256=bh,script_sha256=r.sha(__file__),health_script_sha256=r.sha(health.__file__),environment={k:v for k,v in env.items() if k.startswith(('CK_TILE_GROUPED_GEMM_','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))})
   (folder/'command.json').write_text(json.dumps(rec,indent=2));print('START',num,'/',len(jobs),j['key'],flush=True);t=time.monotonic()
   try:
    with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=min(1800,max(30,int(deadline-time.time()))))
    out=(folder/'run.log').read_text(errors='replace');rec.update(rc=p.returncode,seconds=time.monotonic()-t,ok=p.returncode==0)
    if j['kind']!='regression':
     m=re.search(r'SELECT instance_id=(\d+)',out);rec['selected']=int(m[1]) if m else None
     rec['ok'] &= (rec['selected'] is not None and rec['selected'] not in [5007,5008,5009,5010] if not j['expected'] else rec['selected']==j['expected']) and 'SUMMARY result=PASS' in out
     if j['kind']=='accuracy':
      checks=re.findall(r'ACCURACY backend=ck_vs_blas repeat=(\d+) checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)',out);rec['comparisons']=checks;rec['ok'] &= bool(checks) and all(int(x[2])==0 for x in checks)
     elif j['kind']=='perf':
      m=re.search(r'PERF backend=ck_device_args (.*)',out);rec['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
      rec['ok'] &= bool(m) and bool(re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',out))
     else:rec['perf_files']=[str(p) for p in (folder/'profile').glob('*.perf')];rec['ok'] &= len(rec['perf_files'])==1
   except Exception as ex:rec.update(ok=False,error=repr(ex),seconds=time.monotonic()-t)
   r.append(journal,rec);print('DONE',j['key'],'ok='+str(rec['ok']),'sec='+str(round(rec['seconds'],2)),flush=True)
   if not rec['ok']:raise RuntimeError('STOP inspect '+str(folder/'run.log')+'; no automatic runtime retry')
 print('ALL_DONE',a.suite,flush=True)
if __name__=='__main__':main()
