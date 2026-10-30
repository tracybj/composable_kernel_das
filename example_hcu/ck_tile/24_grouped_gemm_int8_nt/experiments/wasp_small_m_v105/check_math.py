"""M64 native/tails,full-range INT8 against FP64,immutable raw/packed weights."""
from pathlib import Path
import ctypes as C,json,os,sys,resource,gc,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert os.environ['HIP_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
lib=C.CDLL(str(E/'build/v105/libprobe.so'));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p;lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
configs=json.loads((Path(__file__).parent/'candidates.json').read_text());records=[]
profiles=[([64],256,128,True),([64,64],512,384,True),([128],768,640,False),([192],512,896,True),([64]*4,4096,7168,True),([63,65],288,384,False),([0,17,64],288,1280,False),([64],512,384,'wide'),([129],256,384,False),([64],32,128,False),([64]*3,128,640,True),([128],256,512,True)]
for pi,(lengths,n,k,mode) in enumerate(profiles):
 torch.manual_seed(20261030+pi);g=len(lengths);arrays=[];sa=k if mode is True else k+16;sb=k if mode is True else k+32
 if mode=='wide':sa=16384;sb=16400
 for m in lengths:
  a=torch.randint(-128,128,(m,sa),device='cuda',dtype=torch.int8);b=torch.randint(-128,128,(n,sb),device='cuda',dtype=torch.int8);raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32);c=torch.as_strided(raw[16:],(m,n),(n+4,1));ref=(a[:,:k].double()@b[:,:k].double().T).to(torch.int32);arrays.append((a,b,raw,c,ref,a.clone(),b.clone()))
 weights=torch.stack([v[1][:,:k] for v in arrays]);packed=weights.reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous();oldpacked=packed.clone()
 for cfg in configs:
  cid=cfg['id'];ispacked=cfg['packed'];ds=(Desc*g)(*[Desc(a.data_ptr(),packed[j].data_ptr() if ispacked else b.data_ptr(),c.data_ptr(),1,m,n,k,sa,k if ispacked else sb,n+4,0,None,None) for j,(m,(a,b,raw,c,ref,olda,oldb)) in enumerate(zip(lengths,arrays))]);ctx=lib.probe_create(ds,g);assert ctx
  try:
   for a,b,raw,c,ref,olda,oldb in arrays:raw.fill_(123456789)
   torch.cuda.synchronize();ms=float(lib.probe_run(ctx,cid,0,2));assert ms>0,(pi,cid,ms)
   for m,(a,b,raw,c,ref,olda,oldb) in zip(lengths,arrays):
    assert torch.equal(c,ref),(pi,lengths,n,k,cid,int((c!=ref).sum()))
    mask=torch.ones_like(raw,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False);assert bool((raw[mask]==123456789).all());assert torch.equal(a,olda) and torch.equal(b,oldb)
   assert torch.equal(packed,oldpacked)
   records.append(dict(profile=pi,lengths=lengths,N=n,K=k,candidate=cid,status='PASS',full_range_int8=True,FP64=True,guards=True,inputs_unchanged=True));print(json.dumps(records[-1]),flush=True)
  finally:lib.probe_destroy(ctx)
 del arrays,ds,a,b,raw,c,ref,olda,oldb,mask,packed,oldpacked,weights
 gc.collect();torch.cuda.empty_cache()
assert len(records)==108
report=dict(status='PASS',checks=records,library_sha256=hashlib.sha256((E/'build/v105/libprobe.so').read_bytes()).hexdigest());(E/'results/wasp_small_m_v105_math.json').write_text(json.dumps(report,indent=2)+'\n');print('V105_MATH_PASS',len(records))
