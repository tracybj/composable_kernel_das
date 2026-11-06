"""Isolated full-range gates and balanced paired screening against frozen v145."""
from pathlib import Path
import ctypes as C,json,os,sys,csv,statistics,hashlib
import torch
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(E/'cached_weights/tests'))
from bindings import Weight,init,lib,ok
from test_scaled_api import Group2
assert os.environ['HIP_VISIBLE_DEVICES']=='7'
torch.set_num_threads(4);torch.manual_seed(20261106);torch.backends.cuda.matmul.allow_tf32=False
out=Path(sys.argv[1]);out.mkdir(exist_ok=True)
native=Path(sys.argv[2]);kp=C.CDLL(str(native))
formal=len(sys.argv)>3 and sys.argv[3]=='formal'
variants=[1] if formal else list(range(4))
class Args(C.Structure):
 _fields_=[('a',C.c_void_p),('b',C.c_void_p),('c',C.c_void_p),('sa',C.c_void_p),('sb',C.c_void_p)]+[(x,C.c_int) for x in ['m','n','k','as_','bs','cs','sas','sbs']]
kp.kp_create.argtypes=[C.POINTER(Args),C.c_int,C.c_int,C.c_int];kp.kp_create.restype=C.c_void_p
kp.kp_bench.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_void_p];kp.kp_bench.restype=C.c_float
kp.kp_destroy.argtypes=[C.c_void_p]
stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
checks=0;records=[]
def validate(c,ref,raw,n):
 assert torch.equal(torch.isnan(c.float()),torch.isnan(ref.float()))
 mask=~torch.isnan(ref.float())
 assert torch.equal(c[mask],ref[mask]),int((c[mask]!=ref[mask]).sum())
 span=c.shape[1]*(n+4)
 assert (raw[:,:16]==77).all() and (raw[:,16+span:]==77).all()
 assert (raw[:,16:16+span].view(c.shape[0],c.shape[1],n+4)[:,:,n:]==77).all()
def fixture(g,m,n,k,pad=False):
 a=torch.randint(-128,128,(g,m,k+16*pad),device='cuda',dtype=torch.int8)
 b=torch.randint(-128,128,(g,n,k+32*pad),device='cuda',dtype=torch.int8)
 refs=torch.stack([(a[i,:,:k].double()@b[i,:,:k].double().T).int() for i in range(g)])
 return a,b,refs
def setup(a,b,ref,typ,mode):
 g,m,n=ref.shape;k=ref.shape[-1];dtype=[torch.int32,torch.bfloat16,torch.float16][typ]
 span=m*(n+4);cap=(span+7)//8*8
 raw=torch.full((g,cap+32),77,dtype=dtype,device='cuda');c=raw[:,16:16+span].view(g,m,n+4)
 sa=None if mode=='none' else torch.rand((g,m if mode=='row' else 1),device='cuda')*.003+.001
 sb=None if mode=='none' else torch.rand((g,n),device='cuda')*.003+.001
 if mode=='special':sb[:,:min(n,8)]=torch.tensor([float('nan'),float('inf'),float('-inf'),-0.,0.,1e-35,1e30,-.5],device='cuda')[:min(n,8)]
 factor=(sa if sa is not None else torch.ones((g,1),device='cuda')).unsqueeze(-1)*(sb if sb is not None else torch.ones((g,n),device='cuda')).unsqueeze(1)
 expected=ref if typ==0 else (ref.float()*factor).to(dtype)
 return raw,c,sa,sb,expected
for m,n,k,pad in [(1,9,16,True),(2,13,640,True),(1,16,7168,False),(2,16,8192,False),(1,7,8208,True)]:
 a,b,ref=fixture(2,m,n,k,pad);aa,bb=a.clone(),b.clone()
 for typ,mode in [(0,'none'),(1,'none'),(2,'none'),(1,'row'),(2,'row'),(1,'special'),(2,'special')]:
  raw,c,sa,sb,expected=setup(a,b,ref,typ,mode)
  desc=(Args*2)(*[Args(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),sa[i].data_ptr() if sa is not None else None,sb[i].data_ptr() if sb is not None else None,m,n,k,a.stride(1),b.stride(1),n+4,1 if mode=='row' else 0,1) for i in range(2)])
  for v in range(4):
   p=kp.kp_create(desc,2,v,typ);assert p
   try:assert kp.kp_bench(p,0,1,stream)>0;validate(c[:,:,:n],expected,raw,n);checks+=1
   finally:kp.kp_destroy(p)
  assert torch.equal(a,aa) and torch.equal(b,bb)
print('GPU_GATES_PASS',checks,flush=True)
(out/'gates.json').write_text(json.dumps(dict(status='PASS',checks=checks,library_sha256=hashlib.sha256(native.read_bytes()).hexdigest())))
targets=list(csv.DictReader((E/'docs/targets_max_all_1191.csv').open()))
for t in targets:
 if 'neighbors' in sys.argv:
  if not (int(t['M']) in [1,2] and 4096<=int(t['N'])<=6144 and 7168<=int(t['K'])<=8192 and int(t['G'])>=12):continue
 elif int(t['ID']) not in [1137,1138,1143,1144,1150,1151,1165,1166,1171,1172]:continue
 g,m,n,k=[int(t[x]) for x in ['G','M','N','K']];a,b,ref=fixture(g,m,n,k);aa,bb=a.clone(),b.clone()
 weights=C.c_void_p();wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=k) for i in range(g)])
 ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)))
 for typ in [1,2]:
  raw,c,sa,sb,expected=setup(a,b,ref,typ,'channel')
  desc=(Args*g)(*[Args(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),sa[i].data_ptr(),sb[i].data_ptr(),m,n,k,k,k,n+4,0,1) for i in range(g)])
  gd=(Group2*g)()
  for i in range(g):
   gd[i].struct_size=C.sizeof(Group2);gd[i].abi_version=2;gd[i].a=a[i].data_ptr();gd[i].c=c[i].data_ptr();gd[i].m=m;gd[i].a_row_stride=k;gd[i].c_row_stride=n+4;gd[i].output_type=typ;gd[i].scale_a=sa[i].data_ptr();gd[i].scale_b=sb[i].data_ptr();gd[i].scale_b_mode=1
  plan=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan)))
  bench=lib.cached_i8_bench_create(plan);assert bench
  ps={v:kp.kp_create(desc,g,v,typ) for v in variants};assert all(ps.values())
  samples={v:[] for v in [-1]+variants}
  try:
   for sample in range(3):
    order=[-1]+variants;shift=sample%len(order);order=order[shift:]+order[:shift]
    for v in order:
     warm,repeat=(10,50) if formal else (5,20)
     ms=lib.cached_i8_bench_run(bench,warm,repeat,stream) if v==-1 else kp.kp_bench(ps[v],warm,repeat,stream)
     assert ms>0;validate(c[:,:,:n],expected,raw,n);samples[v].append(ms)
   base=statistics.median(samples[-1])
   for v,ss in samples.items():
    ms=statistics.median(ss);record=dict(ID=int(t['ID']),M=m,N=n,K=k,G=g,output=typ,variant=v,ms=ms,TOPS=2*g*m*n*k/ms/1e9,R=2*g*m*n*k/ms/1e9/float(t['target_TOPS']),speedup=base/ms,samples=ss)
    records.append(record);print(json.dumps(record),flush=True)
   assert torch.equal(a,aa) and torch.equal(b,bb)
  finally:
   for p in ps.values():kp.kp_destroy(p)
   lib.cached_i8_bench_destroy(bench);ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
 ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
 (out/'records.json').write_text(json.dumps(records,indent=2))
(out/'done.json').write_text(json.dumps(dict(status='PASS',gates=checks,records=len(records))))
