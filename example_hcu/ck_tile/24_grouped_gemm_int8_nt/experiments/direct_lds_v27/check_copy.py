"""Validate direct-to-LDS linear layout, XOR row layout and OOB fill semantics."""
from pathlib import Path
import ctypes as C,json,os
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
E=Path(__file__).resolve().parents[2]
lib=C.CDLL(str(E/'build/direct_lds_v27/libcopy.so'))
lib.direct_lds_copy_probe.argtypes=[C.c_void_p,C.c_void_p,C.c_int];lib.direct_lds_copy_probe.restype=C.c_int
source=torch.arange(256,device='cuda',dtype=torch.int32);results=[]
for mode in (0,1,2):
 raw=torch.full((288,),-1234567,device='cuda',dtype=torch.int32);out=raw[16:-16]
 rc=lib.direct_lds_copy_probe(source.data_ptr(),out.data_ptr(),mode);assert rc==0,rc
 expected=source.clone()
 if mode==2:expected.reshape(64,4)[1::2]=0
 passed=torch.equal(out,expected)
 guard=bool((raw[:16]==-1234567).all()&(raw[-16:]==-1234567).all())
 rec=dict(mode=mode,equal=passed,guard=guard,head=out[:32].tolist());results.append(rec);print(json.dumps(rec),flush=True)
 assert passed and guard,rec
(E/'results/direct_lds_copy_v27.json').write_text(json.dumps(results,indent=2))
