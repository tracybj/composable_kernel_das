from pathlib import Path
import os,json,ctypes as C
import torch
assert os.environ['HIP_VISIBLE_DEVICES']=='7';torch.set_num_threads(4)
e=Path(__file__).resolve().parents[2];lib=C.CDLL(str(e/'build/v61/libdiag.so'));lib.diag_run.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int];lib.diag_run.restype=C.c_int
rows=[]
for padded in [False,True]:
 torch.manual_seed(6060);sa=272 if padded else 256;sb=288 if padded else 256
 a=torch.randint(-128,128,(256,sa),device='cuda',dtype=torch.int8);b=torch.randint(-128,128,(256,sb),device='cuda',dtype=torch.int8);aa=a.clone();bb=b.clone()
 ref=(a[:,:256].double()@b[:,:256].double().T).to(torch.int32)
 for mode in [3,0,1,2]:
  raw=torch.full((65536+32,),123456789,device='cuda',dtype=torch.int32);out=raw[16:-16].view(256,256)
  rc=lib.diag_run(a.data_ptr(),b.data_ptr(),out.data_ptr(),sa,sb,mode);assert rc==0,rc
  expected=[a[:,:256].int(),b[:,:256].int(),ref,torch.arange(768,device='cuda',dtype=torch.int32)][mode]
  got=out.flatten()[:768] if mode==3 else out
  bad=int((got!=expected).sum());guard=bool((raw[:16]==123456789).all()&(raw[-16:]==123456789).all());same=torch.equal(a,aa) and torch.equal(b,bb)
  coords=(got!=expected).nonzero();
  x=dict(first_bad_coords=coords[:12].tolist(),bad_rows=sorted(set(coords[:,0].tolist())) if mode!=3 else [],padded=padded,mode=mode,bad=bad,guard=guard,inputs_unchanged=same,got=got.flatten()[:16].tolist(),expected=expected.flatten()[:16].tolist());rows.append(x);print(json.dumps(x),flush=True);assert guard and same
(e/'results/v61_io_diagnostic.json').write_text(json.dumps(rows,indent=2))
