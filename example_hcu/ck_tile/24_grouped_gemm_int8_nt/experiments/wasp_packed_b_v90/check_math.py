"""Packed-layout diagnostic;full-range INT8/FP64,guards,full input clones."""
from pathlib import Path
import ctypes as C,json,os,sys,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);E=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
lib=C.CDLL(str(E/'build/v90/libprobe.so'))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
cfg=json.loads((Path(__file__).parent/'candidates.json').read_text())
def pack(x):
 m,k=x.shape;assert m%16==0 and k%64==0
 return x.reshape(m//16,16,k//64,4,16).permute(2,0,3,1,4).contiguous()
cases=[([256],256,k,True) for k in [128,256,384,640,896,1024,3072,6144,7168]]+[
 ([256],512,512,False),([512],256,1024,False),([256,512],512,384,True),
 ([16],256,128,True),([128],256,384,False),([0,256],512,256,True),([256],256,16384,True)]
records=[]
for lengths,n,k,contig in cases:
 torch.manual_seed(20261029+sum(lengths)+k);arrays=[]
 for m in lengths:
  a=torch.randint(-128,128,(m,k if contig else k+16),device='cuda',dtype=torch.int8)
  b=torch.randint(-128,128,(n,k if contig else k+32),device='cuda',dtype=torch.int8)
  pa,pb=pack(a[:,:k]),pack(b[:,:k]);raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32)
  c=torch.as_strided(raw[16:],(m,n),(n+4,1));ref=(a[:,:k].double()@b[:,:k].double().T).to(torch.int32)
  arrays.append((a,b,pa,pb,raw,c,ref,a.clone(),b.clone(),pa.clone(),pb.clone()))
 for config in cfg:
  candidate=config['candidate'];packed=config['packed_B'];trans=config['transpose'];descs=[]
  for m,z in zip(lengths,arrays):
   a,b,pa,pb,raw,c,*_=z;raw.fill_(123456789)
   ap=pa if packed and trans else a;bp=pb if packed and not trans else b
   descs.append(Desc(ap.data_ptr(),bp.data_ptr(),c.data_ptr(),1,m,n,k,k if packed and trans else a.stride(0),k if packed and not trans else b.stride(0),n+4,0,None,None))
  ds=(Desc*len(descs))(*descs);ctx=lib.probe_create(ds,len(descs));assert ctx
  try:
   torch.cuda.synchronize();ms=float(lib.probe_run(ctx,candidate,0,2));assert ms>0
   bad=0
   for m,z in zip(lengths,arrays):
    a,b,pa,pb,raw,c,ref,oa,ob,opa,opb=z;bad+=int((c!=ref).sum())
    mask=torch.ones_like(raw,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False)
    assert bool((raw[mask]==123456789).all())
    assert all(torch.equal(x,y) for x,y in [(a,oa),(b,ob),(pa,opa),(pb,opb)])
   assert bad==0,(lengths,n,k,config,bad)
   rec=dict(lengths=lengths,N=n,K=k,candidate=candidate,status='PASS',bad_elements=bad,guard=True,input_unchanged=True,full_range_int8=True,fp64_reference=True,packed_B=packed,transpose=trans,contiguous_inputs=contig)
   records.append(rec);print(json.dumps(rec),flush=True)
  finally:lib.probe_destroy(ctx)
(E/'results/d2l_v90_math.json').write_text(json.dumps(records,indent=2)+'\n')
