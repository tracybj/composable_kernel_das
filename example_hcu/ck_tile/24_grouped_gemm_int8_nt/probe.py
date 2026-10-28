"""Candidate probe, not a public-interface acceptance result."""
from pathlib import Path
import ctypes as C,os,sys,json,csv,statistics,gc,time
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from bench_int8_nt import parse_shapes
from worker import Desc
cases=parse_shapes('/workspace/GroupGemm/baseline/moe_grouped_gemm_tuning_shape_final.md')
libpath=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'build/libint8_nt_probe.so'
lib=C.CDLL(str(libpath))
lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float
lib.probe_destroy.argtypes=[C.c_void_p]
ids=[int(s) for s in (sys.argv[1] if len(sys.argv)>1 else '1,5,6,9,35,891,1158').split(',')]
dest=ROOT/'results'/(sys.argv[3] if len(sys.argv)>3 else 'probe_manual');dest.mkdir(parents=True,exist_ok=True)
print('DEVICE',torch.cuda.get_device_name(),torch.cuda.mem_get_info(),flush=True)
with (dest/'records.jsonl').open('a') as out:
 for idx in ids:
    s=cases[idx-1];g,m,n,k=[s[key] for key in ['G','M','N','K']]
    print('CASE',idx,m,n,k,g,flush=True)
    torch.manual_seed(20261026+idx)
    a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8)
    b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
    raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32)
    c=raw[:,16:-16].view(g,m,n)
    ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
    ctx=lib.probe_create(ds,g);assert ctx
    variants=[int(v) for v in sys.argv[4].split(',')] if len(sys.argv)>4 else [6004,6005,6006,6007,6008]
    for candidate in variants:
        c.fill_(123456789);torch.cuda.synchronize()
        ts=[float(lib.probe_run(ctx,candidate,2,5)) for _ in range(3)]
        if min(ts)<=0:raise RuntimeError((idx,candidate,ts))
        bad=0
        for e in range(g):
            ref=(a[e].float()@b[e].float().T).to(torch.int32)
            bad+=int((c[e]!=ref).sum())
            del ref
        guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
        rec=dict(id=idx,M=m,N=n,K=k,E=g,candidate=candidate,ms=statistics.median(ts),samples_ms=ts,tops=2*g*m*n*k/(statistics.median(ts)*1e9),bad_elements=bad,guard=guard)
        out.write(json.dumps(rec)+'\n');out.flush();print(json.dumps(rec),flush=True)
        if bad or not guard:raise RuntimeError('Correctness failure')
    lib.probe_destroy(ctx);del a,b,c,raw,ds;gc.collect();torch.cuda.empty_cache()
