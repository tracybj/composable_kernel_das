"""Frozen BLAS2979 and public CK on identical M16384/N4096/K6144/G8 buffers."""
from pathlib import Path
import ctypes as C,csv,json,hashlib,os,sys,statistics,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7' and os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];profile=os.environ.get('CK_PROFILE')=='1'
out=Path(os.environ.get('CK_PROFILE_OUTPUT',str(E/'results/fresh_matched_v75')));out.mkdir(exist_ok=profile)
base=Path('/workspace/GroupGemm/baseline/blaslt_tuning_tools');sys.path.insert(0,str(base))
from tuning_common import Native,Desc as BlasDesc
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc as CKDesc
cfg=json.loads((base/'results/int8_nt_dual_20261027_191653/matmul/run_config.json').read_text())
assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==cfg['binary_sha256']
assert hashlib.sha256(Path(cfg['blas_library']).read_bytes()).hexdigest()==cfg['blas_sha256']
case=next(x for x in cfg['cases'] if x['id']==1038);g,m,n,k=[int(case[x]) for x in ['G','M','N','K']];assert (g,m,n,k)==(8,16384,4096,6144)
target=next(x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open()) if int(x['ID'])==1038)
libpath=E/'build/public_v73/libint8_nt_public.so';identity=hashlib.sha256(libpath.read_bytes()).hexdigest();assert identity=='cc5388e0e0d731a9775b8e4b2c4e8576ca62a4a284bc0f2cbdcd7449d086d16b'
torch.manual_seed(2026102975)
a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8);old_a=a.clone();old_b=b.clone()
raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
reference=torch.empty_like(c)
for i in range(g):reference[i]=(a[i].float()@b[i].float().T).to(torch.int32)
blas_ds=(BlasDesc*g)(*[BlasDesc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr()) for i in range(g)])
ck_ds=(CKDesc*g)(*[CKDesc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
native=Native(cfg,case,blas_ds);selected=[x for x in native.candidates if x['algorithm_id']==2979];assert len(selected)==1;selected=selected[0]
assert 'structbuffer1024' in selected['kernel_name'] and 'MT256x256x128' in selected['kernel_name']
ck=C.CDLL(str(libpath));ck.int8_create.argtypes=[C.POINTER(CKDesc),C.c_int];ck.int8_create.restype=C.c_void_p;ck.int8_run.argtypes=[C.c_void_p,C.c_int,C.c_int];ck.int8_run.restype=C.c_float;ck.int8_destroy.argtypes=[C.c_void_p];ck.int8_instance.argtypes=[C.c_void_p];ck.int8_instance.restype=C.c_uint
ctx=ck.int8_create(ck_ds,g);assert ctx and ck.int8_instance(ctx)==19003
warm,repeat,samples=(2,1,1) if profile else (5,20,3)
manifest=dict(M=m,N=n,K=k,G=g,gpu=7,dtype='int8_int32',layout='NT',scale=None,ck_library_sha256=identity,blas_library_sha256=cfg['blas_sha256'],bridge_sha256=cfg['binary_sha256'],algorithm_id=2979,ck_instance=19003,blas_kernel=selected['kernel_name'],warmup=warm,repeat=repeat,samples=samples,profiler=profile,scope='same physical allocations; frozen algorithm; no retuning; all8 independent groups included',target_TOPS=float(target['target_TOPS']),target_ms=float(target['target_ms']))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'selected.json').write_text(json.dumps(selected,indent=2))
records=[]
try:
 with (out/'records.jsonl').open('w') as f:
  for sample in range(samples):
   order=['BLAS','CK'] if sample%2==0 else ['CK','BLAS']
   for backend in order:
    c.fill_(123456789);torch.cuda.synchronize()
    ms=native.time(selected['candidate_index'],blas_ds,warm,repeat) if backend=='BLAS' else float(ck.int8_run(ctx,warm,repeat));assert ms>0
    bad=0
    for i in range(g):
     for row in range(0,m,256):bad+=int((c[i,row:row+256]!=reference[i,row:row+256]).sum())
    guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=torch.equal(a,old_a) and torch.equal(b,old_b);assert bad==0 and guard and unchanged
    rec=dict(sample=sample,backend=backend,ms=ms,TOPS=2*g*m*n*k/ms/1e9,R=float(target['target_ms'])/ms,bad_elements=bad,guard=guard,full_inputs_unchanged=unchanged)
    records.append(rec);f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
 summary=[]
 for backend in ['BLAS','CK']:
  ts=[x['ms'] for x in records if x['backend']==backend];med=statistics.median(ts);summary.append(dict(backend=backend,ms=med,TOPS=2*g*m*n*k/med/1e9,R=float(target['target_ms'])/med,samples_ms=ts))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'done.json').write_text(json.dumps(dict(status='PASS',records=len(records))))
 print('SUMMARY',json.dumps(summary),flush=True)
finally:ck.int8_destroy(ctx);native.close()
