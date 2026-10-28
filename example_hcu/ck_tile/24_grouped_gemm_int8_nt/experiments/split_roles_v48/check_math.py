"""Full-range INT8/FP64 oracle for D2L K permutation, tails and role barriers."""
from pathlib import Path
import ctypes as C,json,os,sys,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
lib=C.CDLL(str(E/'build/v48/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
records=[]
for lengths,n,k in [([256],256,256),([1],256,256),([257,259],260,528),([128],256,1024),([256],512,4096),([256],256,256),([256,512],512,512),([512],256,1024),([512],512,4096)]:
 torch.manual_seed(20261028+sum(lengths)+k)
 arrays=[];descs=[]
 for m in lengths:
  a=torch.randint(-128,128,(m,k+16),device='cuda',dtype=torch.int8)
  b=torch.randint(-128,128,(n,k+32),device='cuda',dtype=torch.int8)
  raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32)
  c=torch.as_strided(raw[16:],(m,n),(n+4,1))
  ref=(a[:,:k].double()@b[:,:k].double().T).to(torch.int32)
  arrays.append((a,b,raw,c,ref,a.clone(),b.clone()))
  descs.append(Desc(a.data_ptr(),b.data_ptr(),c.data_ptr(),1,m,n,k,k+16,k+32,n+4,0,None,None))
 ds=(Desc*len(lengths))(*descs);ctx=lib.probe_create(ds,len(lengths));assert ctx
 try:
  for candidate in (9001, 9002, 9003, 9004, 9005, 9006):
   for a,b,raw,c,ref,old_a,old_b in arrays:raw.fill_(123456789)
   torch.cuda.synchronize()
   ms=float(lib.probe_run(ctx,candidate,0,2));assert ms>0,(candidate,ms)
   for m,(a,b,raw,c,ref,old_a,old_b) in zip(lengths,arrays):
    bad=int((c!=ref).sum())
    mask=torch.ones_like(raw,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False)
    guard=bool((raw[mask]==123456789).all())
    unchanged=torch.equal(a,old_a) and torch.equal(b,old_b)
    assert bad==0 and guard and unchanged,(lengths,n,k,candidate,bad,guard,unchanged,c[:2,:8].tolist(),ref[:2,:8].tolist())
   rec=dict(lengths=lengths,N=n,K=k,candidate=candidate,status='PASS',full_range_int8=True,fp64_reference=True,guard=True,input_unchanged=True)
   records.append(rec);print(json.dumps(rec),flush=True)
 finally:lib.probe_destroy(ctx)
(E/'results/d2l_v48_math.json').write_text(json.dumps(records,indent=2))
