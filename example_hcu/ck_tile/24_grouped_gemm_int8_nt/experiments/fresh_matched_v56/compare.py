"""Fresh, no-retuning BLASLt/CK comparison on identical physical allocations."""
from pathlib import Path
import ctypes as C,csv,json,hashlib,os,sys,statistics,resource,subprocess
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7' and os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];out=E/'results/fresh_matched_v56';out.mkdir(exist_ok=False)
base=Path('/workspace/GroupGemm/baseline/blaslt_tuning_tools');sys.path.insert(0,str(base))
from tuning_common import Native,Desc as BlasDesc
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc as CKDesc
run=base/'results/int8_nt_dual_20261027_191653/groupgemm'
cfg=json.loads((run/'run_config.json').read_text());cfg['binary']=str(run/'bin/libgroupgemm_tune.so')
assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==cfg['binary_sha256']
case=next(x for x in cfg['cases'] if x['id']==1177)
g,m,n,k=[int(case[x]) for x in ['G','M','N','K']]
assert (g,m,n,k)==(128,2048,4096,8192)
target=next(x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open()) if int(x['ID'])==1177)
assert all(int(target[x])==v for x,v in [('G',g),('M',m),('N',n),('K',k)])
prior=[json.loads(x) for x in (E/'results/hoisted_copy_v55/records.jsonl').read_text().splitlines() if json.loads(x)['id']==1177]
best_t=min([x for x in prior if 9501<=x['candidate']<=9506],key=lambda x:x['ms'])['candidate']
best_n=min([x for x in prior if x['candidate'] in [9507,9508]],key=lambda x:x['ms'])['candidate']
ids=[9003,best_t,best_n];libpath=E/'build/v55/libprobe.so';identity=hashlib.sha256(libpath.read_bytes()).hexdigest()
assert identity==json.loads((E/'results/hoisted_copy_v55/manifest.json').read_text())['library_sha256']
manifest=dict(M=m,N=n,K=k,G=g,gpu=7,dtype='int8_int32',layout='NT',scale=None,
    ck_library_sha256=identity,frozen_bridge_sha256=cfg['binary_sha256'],algorithm_id=6950,
    ck_candidates=ids,target_TOPS=float(target['target_TOPS']),target_ms=float(target['target_ms']),
    warmup=5,repeat=20,samples=3,scope='Same allocations, original shape, no retuning, alternating order; frozen target unchanged',
    source_revision=subprocess.check_output(['git','-C',str(E.parents[2]),'rev-parse','HEAD'],text=True).strip())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
torch.manual_seed(2026102901)
a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
old_a=a.clone();old_b=b.clone()
raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
reference=torch.empty_like(c)
for group in range(g):reference[group]=(a[group].float()@b[group].float().T).to(torch.int32)
blas_ds=(BlasDesc*g)(*[BlasDesc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr()) for i in range(g)])
ck_ds=(CKDesc*g)(*[CKDesc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
native=Native(cfg,case,blas_ds);ck=C.CDLL(str(libpath))
ck.probe_create.argtypes=[C.POINTER(CKDesc),C.c_int];ck.probe_create.restype=C.c_void_p
ck.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];ck.probe_run.restype=C.c_float
ck.probe_destroy.argtypes=[C.c_void_p]
ctx=ck.probe_create(ck_ds,g);assert ctx
records=[]
try:
 selected=[x for x in native.candidates if x['algorithm_id']==6950];assert len(selected)==1
 selected=selected[0];assert 'GG_AI_SAV_MT256x256x32' in selected['kernel_name']
 (out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')
 paths=[('BLAS',6950)]+[('CK',x) for x in ids]
 with (out/'records.jsonl').open('w') as f:
  for sample in range(3):
   order=paths if sample%2==0 else paths[::-1]
   for backend,ident in order:
    (out/'active.json').write_text(json.dumps(dict(sample=sample,backend=backend,candidate=ident)))
    c.fill_(123456789);torch.cuda.synchronize()
    ms=float(native.time(selected['candidate_index'],blas_ds,5,20)) if backend=='BLAS' else float(ck.probe_run(ctx,ident,5,20))
    assert ms>0
    bad=0
    for group in range(g):bad+=int((c[group]!=reference[group]).sum())
    guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
    unchanged=torch.equal(a,old_a) and torch.equal(b,old_b)
    assert bad==0 and guard and unchanged,(backend,ident,bad,guard,unchanged)
    row=dict(sample=sample,backend=backend,candidate=ident,ms=ms,TOPS=2*g*m*n*k/ms/1e9,
        R_to_frozen=float(target['target_ms'])/ms,bad_elements=bad,guard=guard,full_inputs_unchanged=unchanged)
    records.append(row);f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
 summary=[]
 for backend,ident in paths:
  ts=[x['ms'] for x in records if x['backend']==backend and x['candidate']==ident];assert len(ts)==3
  med=statistics.median(ts);summary.append(dict(backend=backend,candidate=ident,samples_ms=ts,median_ms=med,TOPS=2*g*m*n*k/med/1e9,R_to_frozen=float(target['target_ms'])/med))
 assert hashlib.sha256(libpath.read_bytes()).hexdigest()==identity
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'done.json').write_text(json.dumps(dict(status='PASS',records=len(records),note='Fixed algorithm replay, no target replacement'))+'\n')
 print('FRESH_MATCHED_SUMMARY',json.dumps(summary),flush=True)
finally:
 ck.probe_destroy(ctx);native.close()
