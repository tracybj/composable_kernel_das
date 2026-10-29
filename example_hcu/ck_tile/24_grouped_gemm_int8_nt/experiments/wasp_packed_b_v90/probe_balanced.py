"""Confirm raw/packed screening gains with rotating candidate order on same allocations."""
from pathlib import Path
import ctypes as C,json,sys,statistics,gc,os,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
from bench_int8_nt import parse_shapes
shapes=parse_shapes('/workspace/GroupGemm/baseline/moe_grouped_gemm_tuning_shape_final.md');out=E/'results/wasp_packed_b_v90'
first=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()];assert len(first)==42
configs={x['candidate']:x for x in json.loads((Path(__file__).parent/'candidates.json').read_text())}
lib=C.CDLL(str(E/'build/v90/libprobe.so'));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
assert not (out/'balanced_records.jsonl').exists()
def pack(x):
 g,m,k=x.shape;return x.view(g,m//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous()
with (out/'balanced_records.jsonl').open('w') as f:
 for idx in [641,698,902,1038,1158,1177,48]:
  s=shapes[idx-1];g,m,n,k=[s[x] for x in ['G','M','N','K']];torch.manual_seed(20261026+idx)
  a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8);pa,pb=pack(a),pack(b)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
  part=[x for x in first if x['id']==idx];ids=[]
  for trans in [False,True]:
   ids.append(min((x for x in part if x['transpose']==trans and not x['packed_B']),key=lambda x:x['ms'])['candidate'])
   ids.append(32101 if trans else 32001)
  contexts={};descs=[];ts={i:[] for i in ids}
  for candidate in ids:
   cfg=configs[candidate];packed,trans=cfg['packed_B'],cfg['transpose'];ap=pa if packed and trans else a;bp=pb if packed and not trans else b
   ds=(Desc*g)(*[Desc(ap[e].data_ptr(),bp[e].data_ptr(),c[e].data_ptr(),1,m,n,k,k,k,n,0,None,None) for e in range(g)]);descs.append(ds)
   contexts[candidate]=lib.probe_create(ds,g);assert contexts[candidate]
  try:
   for rnd in range(4):
    order=ids[rnd:]+ids[:rnd]
    for candidate in order:
     torch.cuda.synchronize();ms=float(lib.probe_run(contexts[candidate],candidate,2,10));assert ms>0;ts[candidate].append(ms)
   for candidate in ids:
    c.fill_(123456789);torch.cuda.synchronize();assert float(lib.probe_run(contexts[candidate],candidate,0,1))>0
    bad=0
    for e in range(g):
     ref=(a[e].float()@b[e].float().T).to(torch.int32);bad+=int((c[e]!=ref).sum());del ref
    guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());assert bad==0 and guard
    ms=statistics.median(ts[candidate]);rec=dict(id=idx,M=m,N=n,K=k,E=g,**configs[candidate],ms=ms,samples_ms=ts[candidate],tops=2*g*m*n*k/(ms*1e9),bad_elements=bad,guard=guard,method='same-allocation rotating4-candidate order;warmup2/repeat10;4samples;median;packing excluded')
    f.write(json.dumps(rec)+'\n');f.flush();print(json.dumps(rec),flush=True)
  finally:
   for ctx in contexts.values():lib.probe_destroy(ctx)
  del a,b,pa,pb,c,raw,ds,descs;gc.collect();torch.cuda.empty_cache()
print('V90_BALANCED_DONE',flush=True)
