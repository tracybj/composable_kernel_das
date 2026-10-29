"""Same-allocation raw/packed GEMM with separately timed GPU packing cost."""
from pathlib import Path
import ctypes as C,json,csv,sys,statistics,gc,os
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
from bench_int8_nt import parse_shapes
shapes=parse_shapes('/workspace/GroupGemm/baseline/moe_grouped_gemm_tuning_shape_final.md')
cfg=json.loads((Path(__file__).parent/'candidates.json').read_text())
lib=C.CDLL(str(E/'build/v90/libprobe.so'));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
out=E/'results/wasp_packed_b_v90';out.mkdir(exist_ok=True);assert not (out/'records.jsonl').exists()
def packing(x):
 g,m,k=x.shape;v=x.view(g,m//16,16,k//64,4,16).permute(0,3,1,4,2,5)
 assert v.data_ptr()==x.data_ptr();dst=torch.empty(v.shape,device='cuda',dtype=torch.int8)
 for _ in range(2):dst.copy_(v)
 ts=[]
 for _ in range(3):
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
  for _ in range(5):dst.copy_(v)
  end.record();end.synchronize();ts.append(start.elapsed_time(end)/5)
 return dst,dict(ms=statistics.median(ts),samples_ms=ts,method='preallocated GPU permuted-view copy;HIP events;warmup2/repeat5/sample3;allocation excluded',bytes=dst.numel())
with (out/'records.jsonl').open('w') as f:
 for idx in [1158,1177,902,1038,698,641,48]:
  s=shapes[idx-1];g,m,n,k=[s[x] for x in ['G','M','N','K']];torch.manual_seed(20261026+idx)
  a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  pa,ta=packing(a);pb,tb=packing(b)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
  for config in cfg:
   packed,trans=config['packed_B'],config['transpose'];ap=pa if packed and trans else a;bp=pb if packed and not trans else b
   ds=(Desc*g)(*[Desc(ap[e].data_ptr(),bp[e].data_ptr(),c[e].data_ptr(),1,m,n,k,k,k,n,0,None,None) for e in range(g)])
   ctx=lib.probe_create(ds,g);assert ctx
   try:
    c.fill_(123456789);torch.cuda.synchronize();ts=[float(lib.probe_run(ctx,config['candidate'],2,5)) for _ in range(3)];assert min(ts)>0
    bad=0
    for e in range(g):
     ref=(a[e].float()@b[e].float().T).to(torch.int32);bad+=int((c[e]!=ref).sum());del ref
    guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());assert bad==0 and guard
    ms=statistics.median(ts);cost=(ta if trans else tb) if packed else dict(ms=0,samples_ms=[],bytes=0,method='raw NT;no conversion')
    rec=dict(id=idx,M=m,N=n,K=k,E=g,**config,ms=ms,samples_ms=ts,tops=2*g*m*n*k/(ms*1e9),bad_elements=bad,guard=guard,packing=cost,one_use_ms=ms+cost['ms'],one_use_TOPS=2*g*m*n*k/((ms+cost['ms'])*1e9))
    f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
   finally:lib.probe_destroy(ctx)
  del a,b,pa,pb,c,raw,ds;gc.collect();torch.cuda.empty_cache()
