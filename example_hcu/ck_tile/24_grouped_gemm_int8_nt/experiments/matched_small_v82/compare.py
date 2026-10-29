"""Paired public v73/v80 diagnostic; preserve both formal result snapshots."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,statistics,resource,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ['HIP_VISIBLE_DEVICES']=='7' and os.environ['CUDA_VISIBLE_DEVICES']=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
out=E/'results/matched_small_v82';out.mkdir();targets={int(x['ID']):x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open())}
expected={73:'cc5388e0e0d731a9775b8e4b2c4e8576ca62a4a284bc0f2cbdcd7449d086d16b',80:'10f21a6d71ed11497c9a3c72b621d8e7a717cb3c7c7294fdc009f2576e1ee04e'}
libs={}
for v,sha in expected.items():
 path=E/f'build/public_v{v}/libint8_nt_public.so';assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 lib=C.CDLL(str(path));lib.int8_create.argtypes=[C.POINTER(Desc),C.c_int];lib.int8_create.restype=C.c_void_p
 lib.int8_run.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.int8_run.restype=C.c_float;lib.int8_destroy.argtypes=[C.c_void_p];lib.int8_instance.argtypes=[C.c_void_p];lib.int8_instance.restype=C.c_uint
 libs[v]=lib
(out/'manifest.json').write_text(json.dumps(dict(gpu=7,library_sha256=expected,shape_ids=[508,1022],warmup=10,repeat=50,samples=3,scope='same physical allocations, alternate v73/v80 order; diagnostic only, formal1178 records unchanged'),indent=2))
records=[]
with (out/'records.jsonl').open('w') as f:
 for idx in [508,1022]:
  t=targets[idx];m,n,k,g=[int(t[x]) for x in ['M','N','K','G']];torch.manual_seed(20261029+idx)
  a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8);oa=a.clone();ob=b.clone()
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n);ref=torch.stack([(a[i].float()@b[i].float().T).to(torch.int32) for i in range(g)])
  ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
  contexts={v:lib.int8_create(ds,g) for v,lib in libs.items()};assert all(contexts.values())
  try:
   for sample in range(3):
    for v in ([73,80] if sample%2==0 else [80,73]):
     lib=libs[v];ctx=contexts[v];c.fill_(123456789);torch.cuda.synchronize();ms=float(lib.int8_run(ctx,10,50));assert ms>0
     bad=int((c!=ref).sum());guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=torch.equal(a,oa) and torch.equal(b,ob);assert bad==0 and guard and unchanged
     x=dict(ID=idx,M=m,N=n,K=k,G=g,version=v,sample=sample,candidate=int(lib.int8_instance(ctx)),ms=ms,TOPS=2*g*m*n*k/ms/1e9,R=float(t['target_ms'])/ms,bad_elements=bad,guard=guard,full_inputs_unchanged=unchanged)
     records.append(x);f.write(json.dumps(x)+'\n');f.flush();print(json.dumps(x),flush=True)
  finally:
   for v,ctx in contexts.items():libs[v].int8_destroy(ctx)
summary=[]
for idx in [508,1022]:
 for v in [73,80]:
  a=[x for x in records if x['ID']==idx and x['version']==v];ms=statistics.median(x['ms'] for x in a);t=targets[idx];m,n,k,g=[int(t[x]) for x in ['M','N','K','G']]
  summary.append(dict(ID=idx,version=v,ms=ms,TOPS=2*g*m*n*k/ms/1e9,R=float(t['target_ms'])/ms,samples_ms=[x['ms'] for x in a]))
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'done.json').write_text(json.dumps(dict(status='PASS',records=len(records))));print('SUMMARY',json.dumps(summary))
