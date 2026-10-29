"""Screen all M>=256 target shapes; explicit candidates are development-only."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,statistics,gc,time,resource,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
assert os.environ['HIP_VISIBLE_DEVICES']=='7' and os.environ['CUDA_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
cases=[x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open()) if int(x['M'])>=256]
formal={x['ID']:x for x in csv.DictReader((E/'docs/current_public_v47_1178.csv').open(encoding='utf-8-sig'))}
out=E/'results/wasp_one_barrier_v69_large_screen';out.mkdir(exist_ok=True)
assert not (out/'records.jsonl').exists(),'Use a fresh run directory'
resume_count=0;total_cases=len(cases)
libpath=E/'build/v69/libprobe.so';identity=hashlib.sha256(libpath.read_bytes()).hexdigest()
assert identity=='a46405b9917e756c344deb335089178201083976e0922fa56f329381ce023f5a'
assert total_cases==344
free,total=torch.cuda.mem_get_info();print('DEVICE',torch.cuda.get_device_name(),free,total,flush=True)
lib=C.CDLL(str(libpath));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
manifest=dict(library_sha256=identity,total=344,gpu=7,warmup=2,repeat=5,samples=3,scope='Original344 M>=256 shapes; six one-barrier A-prefetch candidates plus current-default instance control, not public acceptance',validation='Every output checked in256-row chunks; inputs[-8,8],TF32 disabled, guard checks and sampled input immutability',source_revision=__import__('subprocess').check_output(['git','-C',str(E.parents[2]),'rev-parse','HEAD'],text=True).strip())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
records=[]
with (out/'records.jsonl').open('w') as stream:
 for case in cases:
  g,m,n,k=[int(case[x]) for x in ('G','M','N','K')];idx=int(case['ID'])
  variants=[int(formal[str(idx)]['instance_id']),18201,18211,18401,18411,18801,18811]
  shift=len(records)//7%len(variants);variants=variants[shift:]+variants[:shift]
  (out/'active.json').write_text(json.dumps(dict(ID=idx,M=m,N=n,K=k,G=g)))
  free_now,_=torch.cuda.mem_get_info()
  estimate=g*(m*k+n*k+8*m*n)+4*(m*k+n*k+m*n)
  assert estimate+1024**3<free_now,('Memory budget insufficient; do not change G',idx,estimate,free_now)
  torch.manual_seed(20261028+idx)
  a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8)
  b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
  reference=torch.empty_like(c)
  for group in range(g):reference[group]=(a[group].float()@b[group].float().T).to(torch.int32)
  samples=[]
  for v in (a,b):
   indices=torch.arange(256,device='cuda',dtype=torch.int64)*(v.numel()-1)//255
   samples.append((indices,v.reshape(-1)[indices].clone()))
  ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
  ctx=lib.probe_create(ds,g);assert ctx
  try:
   for candidate in variants:
    c.fill_(123456789);torch.cuda.synchronize()
    ts=[float(lib.probe_run(ctx,candidate,2,5)) for _ in range(3)]
    assert min(ts)>0
    bad=0
    for group in range(g):
     for row0 in range(0,m,256):bad+=int((c[group,row0:row0+256]!=reference[group,row0:row0+256]).sum())
    guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
    unchanged=all(torch.equal(v.reshape(-1)[indices],old) for v,(indices,old) in zip((a,b),samples))
    assert bad==0 and guard and unchanged,(case,candidate,bad,guard,unchanged)
    ms=statistics.median(ts)
    rec=dict(ID=idx,M=m,N=n,K=k,G=g,candidate=candidate,ms=ms,samples_ms=ts,TOPS=2*g*m*n*k/ms/1e9,target_TOPS=float(case['target_TOPS']),R=float(case['target_ms'])/ms,bad_elements=bad,guard=guard,input_samples_unchanged=unchanged)
    stream.write(json.dumps(rec)+'\n');stream.flush();records.append(rec)
  finally:lib.probe_destroy(ctx)
  best=min(records[-len(variants):],key=lambda x:x['ms'])
  progress=dict(completed_shapes=resume_count+cases.index(case)+1,total=total_cases,last_ID=idx,best_candidate=best['candidate'],best_R=best['R'])
  (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
  del a,b,c,raw,reference,ds,samples,v,indices;gc.collect();torch.cuda.empty_cache()
assert hashlib.sha256(libpath.read_bytes()).hexdigest()==identity
(out/'done.json').write_text(json.dumps(dict(status='PASS',shapes=total_cases,records=len(records),note='Screening only; final acceptance requires default public selector and independent full-domain validation.')))
print('SCREEN_DONE',flush=True)
