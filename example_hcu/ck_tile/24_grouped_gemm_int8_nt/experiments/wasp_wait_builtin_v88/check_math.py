"""Full-range INT8/FP64 checks for thecompiler-visible current-B wait and early/late prefetch."""
from pathlib import Path
import ctypes as C,json,os,sys,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
lib=C.CDLL(str(E/'build/v88/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
records=[]
for lengths,n,k,contiguous in [(*x,False) for x in [([256],256,256),([1],256,256),([257,259],260,528),([128],256,1024),([256],512,4096),([256],256,256),([256,512],512,512),([512],256,1024),([512],512,4096),([256],260,4112),([257],260,4192)]]+[([256],256,3072,True),([256],256,6144,True),([256],256,7168,True),([256],256,16384,False),([257],260,128,False),([256],256,256,True),([512],512,1024,True),([256],512,4096,True),([512],256,512,True),([256],512,384,True),([512],256,640,True),([512],512,320,True),([1024],512,448,True),([512],512,576,True),([512,1024],512,1024,True),([1024],1024,6144,True),([256],256,128,True),([512],512,896,True),([256],512,1152,True)]:
 torch.manual_seed(20261028+sum(lengths)+k)
 arrays=[];descs=[]
 for m in lengths:
  a=torch.randint(-128,128,(m,k if contiguous else k+16),device='cuda',dtype=torch.int8)
  b=torch.randint(-128,128,(n,k if contiguous else k+32),device='cuda',dtype=torch.int8)
  raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32)
  c=torch.as_strided(raw[16:],(m,n),(n+4,1))
  ref=(a[:,:k].double()@b[:,:k].double().T).to(torch.int32)
  arrays.append((a,b,raw,c,ref,a.clone(),b.clone()))
  descs.append(Desc(a.data_ptr(),b.data_ptr(),c.data_ptr(),1,m,n,k,int(a.stride(0)),int(b.stride(0)),n+4,0,None,None))
 ds=(Desc*len(lengths))(*descs);ctx=lib.probe_create(ds,len(lengths));assert ctx
 try:
  for candidate in (30000, 30001, 30002, 30003, 30010, 30011, 30012, 30013, 30100, 30101, 30102, 30103, 30110, 30111, 30112, 30113):
   for a,b,raw,c,ref,old_a,old_b in arrays:raw.fill_(123456789)
   torch.cuda.synchronize()
   ms=float(lib.probe_run(ctx,candidate,0,2));assert ms>0,(candidate,ms)
   bad_total=0
   for m,(a,b,raw,c,ref,old_a,old_b) in zip(lengths,arrays):
    bad=int((c!=ref).sum())
    mask=torch.ones_like(raw,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False)
    guard=bool((raw[mask]==123456789).all())
    unchanged=torch.equal(a,old_a) and torch.equal(b,old_b)
    assert guard and unchanged,(lengths,n,k,candidate,bad,guard,unchanged)
    bad_total+=bad
   rec=dict(lengths=lengths,N=n,K=k,candidate=candidate,status='PASS' if bad_total==0 else 'FAIL',bad_elements=bad_total,contiguous_inputs=contiguous,full_range_int8=True,fp64_reference=True,guard=True,input_unchanged=True)
   records.append(rec);print(json.dumps(rec),flush=True)
 finally:lib.probe_destroy(ctx)
(E/'results/d2l_v88_math.json').write_text(json.dumps(records,indent=2))
