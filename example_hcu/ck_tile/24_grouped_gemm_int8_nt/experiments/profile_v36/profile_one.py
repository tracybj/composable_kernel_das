"""One existing, numerically validated candidate; not a new acceptance run."""
from pathlib import Path
import ctypes as C,os,sys,json,csv,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4)
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
shape=next(x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open()) if x['ID']=='1177')
g,m,n,k=[int(shape[x]) for x in ['G','M','N','K']]
torch.manual_seed(20261028)
a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8)
b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
c=torch.empty((g,m,n),device='cuda',dtype=torch.int32)
lib=C.CDLL(str(E/'build/v30/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
ctx=lib.probe_create(ds,g);assert ctx
try:
 torch.cuda.synchronize()
 ms=float(lib.probe_run(ctx,7204,0,1));assert ms>0
 bad=0
 for i in range(g):bad+=int((c[i]!=(a[i].float()@b[i].float().T).to(torch.int32)).sum())
 assert bad==0
 print(json.dumps(dict(M=m,N=n,K=k,G=g,candidate=7204,profile_event_ms=ms,bad_elements=bad,scope='profiling; do not compare instrumented timing to baseline')),flush=True)
finally:lib.probe_destroy(ctx)
