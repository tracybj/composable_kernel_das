"""Same allocations: both-operands LDS versus A-only LDS/direct B, ordered MMAC."""
from pathlib import Path
import ctypes as C,os,sys,json,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
g,m,n,k=128,2048,4096,8192;torch.manual_seed(20261028)
a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
old_a=a.clone();old_b=b.clone()
raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
lib=C.CDLL(str(E/'build/v52/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
ctx=lib.probe_create(ds,g);assert ctx;records=[]
try:
 for candidate in [9003,9301,9302]:
  c.fill_(123456789);torch.cuda.synchronize();ms=float(lib.probe_run(ctx,candidate,2,1));assert ms>0
  bad=0
  for i in range(g):bad+=int((c[i]!=(a[i].float()@b[i].float().T).to(torch.int32)).sum())
  guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=torch.equal(a,old_a) and torch.equal(b,old_b)
  assert bad==0 and guard and unchanged
  record=dict(candidate=candidate,M=m,N=n,K=k,G=g,profile_event_ms=ms,warmup=2,repeat=1,bad_elements=bad,output_guards=guard,full_inputs_unchanged=unchanged)
  records.append(record);print(json.dumps(record),flush=True)
finally:lib.probe_destroy(ctx)
(Path(os.environ.get('CK_PROFILE_OUTPUT',str(E/'results/profile_v53_operand_paths')))/'validation.json').write_text(json.dumps(records,indent=2))
