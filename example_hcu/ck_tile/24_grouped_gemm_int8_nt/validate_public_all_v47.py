"""Complete current public-v47 results, reusing only SHA-identical audited records."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,statistics,gc,resource,hashlib,subprocess,datetime
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
assert os.environ['HIP_VISIBLE_DEVICES']=='7' and os.environ['CUDA_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
out=E/'results/public_v47_all_1178';out.mkdir(exist_ok=True)
assert not (out/'records.jsonl').exists(),'Existing run: inspect before resuming'
cases=list(csv.DictReader((E/'docs/targets_int8_nt.csv').open()));assert len(cases)==1178
libpath=E/'build/public_v47/libint8_nt_public.so';identity=hashlib.sha256(libpath.read_bytes()).hexdigest()
assert identity=='865755915e5f4b9045d40689311ca531f7980b7531b9ac0b4cf22748a51e9bb5'
old=E/'results/public_v47_small_m';manifest=json.loads((old/'manifest.json').read_text());assert manifest['library_sha256']==identity
records=[json.loads(s) for s in (old/'records.jsonl').read_text().splitlines()];assert len(records)==560
byid={x['ID']:x for x in records};assert len(byid)==560
for t in cases:
 if int(t['ID']) in byid:
  x=byid[int(t['ID'])];assert int(t['M'])<=16 and all(x[k]==int(t[k]) for k in ['M','N','K','G'])
  assert x['bad_elements']==0 and x['guard'] and x['input_samples_unchanged']
  x.update(measurement_source='public_v47_small_m',warmup=10,repeat=50,samples=3)
fields=['ID','phase','M','N','K','G','dtype','layout','instance_id','CK_ms','CK_TOPS','BLAS_target_ms','BLAS_target_TOPS','R','numeric_status','measurement_source','warmup','repeat','samples','library_sha256']
def export():
 tmp=out/'results.csv.tmp'
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for t in cases:
   x=byid.get(int(t['ID']));d={k:t[k] for k in ['ID','phase','M','N','K','G']}
   d.update(dtype='INT8xINT8->INT32_no_scale',layout='NT',BLAS_target_ms=t['target_ms'],BLAS_target_TOPS=t['target_TOPS'],library_sha256=identity,numeric_status='PENDING')
   if x:d.update(instance_id=x['candidate'],CK_ms=x['ms'],CK_TOPS=x['TOPS'],R=x['R'],numeric_status='PASS',**{k:x[k] for k in ['measurement_source','warmup','repeat','samples']})
   w.writerow(d)
 tmp.replace(out/'results.csv')
export()
(out/'manifest.json').write_text(json.dumps(dict(total=1178,reused=560,new=618,gpu=7,library_sha256=identity,source_revision=subprocess.check_output(['git','-C',str(E.parents[2]),'rev-parse','HEAD'],text=True).strip(),public_implementation_revision='5a0ed2b',new_warmup=10,new_repeat=50,new_samples=3,input_range=[-8,8],validation='Every output against FP32 exact reference, TF32 disabled, bounded chunks; guards and sampled input immutability. Full-range boundary tests separately recorded.',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
lib=C.CDLL(str(libpath));lib.int8_create.argtypes=[C.POINTER(Desc),C.c_int];lib.int8_create.restype=C.c_void_p
lib.int8_run.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.int8_run.restype=C.c_float
lib.int8_destroy.argtypes=[C.c_void_p];lib.int8_instance.argtypes=[C.c_void_p];lib.int8_instance.restype=C.c_uint
with (out/'records.jsonl').open('w') as stream:
 for x in records:stream.write(json.dumps(x)+'\n')
 stream.flush()
 for t in cases:
  idx=int(t['ID'])
  if idx in byid:continue
  g,m,n,k=[int(t[x]) for x in ['G','M','N','K']]
  (out/'active.json').write_text(json.dumps(dict(ID=idx,M=m,N=n,K=k,G=g)))
  torch.manual_seed(20261028+idx)
  a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
  samples=[]
  for v in [a,b]:
   inds=torch.arange(256,device='cuda',dtype=torch.int64)*(v.numel()-1)//255;samples.append((inds,v.reshape(-1)[inds].clone()))
  ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
  ctx=lib.int8_create(ds,g);assert ctx
  try:
   candidate=int(lib.int8_instance(ctx));c.fill_(123456789);torch.cuda.synchronize()
   ts=[float(lib.int8_run(ctx,10,50)) for _ in range(3)];assert min(ts)>0
   bad=0
   for group in range(g):
    bf=b[group].float()
    for start in range(0,m,128):
     ref=(a[group,start:start+128].float()@bf.T).to(torch.int32)
     bad+=int((c[group,start:start+128]!=ref).sum())
    del bf,ref
   guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
   unchanged=all(torch.equal(v.reshape(-1)[inds],oldv) for v,(inds,oldv) in zip((a,b),samples))
   assert bad==0 and guard and unchanged,(idx,candidate,bad,guard,unchanged)
   ms=statistics.median(ts);x=dict(ID=idx,M=m,N=n,K=k,G=g,candidate=candidate,ms=ms,samples_ms=ts,TOPS=2*g*m*n*k/ms/1e9,target_TOPS=float(t['target_TOPS']),R=float(t['target_ms'])/ms,bad_elements=bad,guard=guard,input_samples_unchanged=unchanged,measurement_source='public_v47_all_1178',warmup=10,repeat=50,samples=3)
   stream.write(json.dumps(x)+'\n');stream.flush();byid[idx]=x;export()
   progress=dict(completed=len(byid),total=1178,reused=560,new_completed=len(byid)-560,last_ID=idx)
   (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
  finally:lib.int8_destroy(ctx)
  del a,b,raw,c,ds,samples,v,inds;gc.collect();torch.cuda.empty_cache()
assert len(byid)==1178 and hashlib.sha256(libpath.read_bytes()).hexdigest()==identity
(out/'done.json').write_text(json.dumps(dict(status='PASS',numeric_pass=1178,ge1=sum(x['R']>=1 for x in byid.values()),min_R=min(x['R'] for x in byid.values()),median_R=statistics.median(x['R'] for x in byid.values()))))
print('ALL_1178_DONE',flush=True)
