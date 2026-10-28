"""Aligned dimensions with >2 GiB operand span must use the 64-bit fallback."""
from pathlib import Path
import ctypes as C,json,os,sys,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
assert torch.cuda.mem_get_info()[0]>8*(1<<30)
m=n=k=256;stride=9*(1<<20)
assert (m-1)*stride+k>0x7fffffff
raw_a=torch.zeros(((m-1)*stride+k+32,),device='cuda',dtype=torch.int8)
a=torch.as_strided(raw_a[16:],(m,k),(stride,1));a.copy_(torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8))
b=torch.randint(-128,128,(n,k+32),device='cuda',dtype=torch.int8)
old_a=raw_a.clone();old_b=b.clone()
raw_c=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32)
c=torch.as_strided(raw_c[16:],(m,n),(n+4,1));ref=(a.double()@b[:,:k].double().T).to(torch.int32)
mask=torch.ones_like(raw_c,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False)
lib=C.CDLL(str(E/'build/v39/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
ds=(Desc*1)(Desc(a.data_ptr(),b.data_ptr(),c.data_ptr(),1,m,n,k,stride,k+32,n+4,0,None,None))
ctx=lib.probe_create(ds,1);assert ctx;records=[]
try:
 for candidate in [8401,8402,8403,8404,8405]:
  raw_c.fill_(123456789);torch.cuda.synchronize();assert lib.probe_run(ctx,candidate,0,2)>0
  assert torch.equal(c,ref) and bool((raw_c[mask]==123456789).all())
  assert torch.equal(raw_a,old_a) and torch.equal(b,old_b)
  record=dict(candidate=candidate,M=m,N=n,K=k,stride_A=stride,operand_span=(m-1)*stride+k,status='PASS',full_range_int8=True,fp64_reference=True,full_input_buffers_unchanged=True,output_guards=True)
  records.append(record);print(json.dumps(record),flush=True)
finally:lib.probe_destroy(ctx)
(E/'results/aligned_v39_large_stride.json').write_text(json.dumps(records,indent=2))
