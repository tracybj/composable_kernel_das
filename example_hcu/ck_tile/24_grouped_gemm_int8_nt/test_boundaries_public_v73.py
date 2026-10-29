#!/usr/bin/env python3
"""Public ABI boundary tests, independent full FP64 integer reference."""
from pathlib import Path
import ctypes as C,json,os,sys
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt')
from worker import Desc
lib=C.CDLL(os.environ.get('CK_INT8_LIB',str(ROOT/'build_cmake/libint8_nt_public.so')))
lib.int8_create.argtypes=[C.POINTER(Desc),C.c_int];lib.int8_create.restype=C.c_void_p
lib.int8_run.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.int8_run.restype=C.c_float
lib.int8_destroy.argtypes=[C.c_void_p];lib.int8_error.restype=C.c_char_p
lib.int8_set_stream.argtypes=[C.c_void_p,C.c_void_p]
lib.int8_contract_check.argtypes=[C.c_void_p,C.c_int];lib.int8_contract_check.restype=C.c_int
lib.int8_mismatch_check.argtypes=[C.c_void_p];lib.int8_mismatch_check.restype=C.c_int
records=[]
for lengths,n,k in [([17,17],260,528),([32]*8,260,1232),([32]*33,260,2064),([33,33],260,1232),([64]*8,260,4112),([32]*4,260,4112),([64]*4,260,4112),([1,17,0],1236,1232),([0,0],128,512),([33,0,257],300,528),([128,256],260,256),([128,128],256,512),([256,256],256,2048),([128],256,2048),([257,257],260,2064),([129,129],260,528),([4096],256,2048),([4096],256,4096),([4097],260,1232),([16,16],256,4096),([1],260,256),([2,2],260,768),([3],260,1232),([8]*17,260,2064),([16]*17,260,528),([1],260,4112),([16]*8,260,6160),([3]*8,260,4096),([256],256,384),([512],256,640),([2048],256,768),([2304],256,768),([2304],256,896),([256],512,4096)]:
    torch.manual_seed(20261026+sum(lengths))
    arrays=[];descs=[]
    sa=16384 if lengths==[256] and n==512 and k==4096 else k+16
    sb=16400 if lengths==[256] and n==512 and k==4096 else k+32
    for m in lengths:
        x=torch.randint(-128,128,(m,sa),device='cuda',dtype=torch.int8)
        w=torch.randint(-128,128,(n,sb),device='cuda',dtype=torch.int8)
        raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32)
        y=torch.as_strided(raw[16:],(m,n),(n+4,1))
        arrays.append((x,w,raw,y,x.clone(),w.clone()))
        descs.append(Desc(x.data_ptr(),w.data_ptr(),y.data_ptr(),1,m,n,k,sa,sb,n+4,0,None,None))
    ds=(Desc*len(lengths))(*descs);ctx=lib.int8_create(ds,len(lengths));assert ctx,lib.int8_error().decode()
    try:
        assert lib.int8_mismatch_check(ctx)==-6
        assert [lib.int8_contract_check(ctx,i) for i in range(5)]==[-5,-5,-2,-3,-1]
        stream=torch.cuda.Stream();torch.cuda.synchronize();lib.int8_set_stream(ctx,stream.cuda_stream)
        assert lib.int8_run(ctx,0,3)>0,lib.int8_error().decode();stream.synchronize()
        for m,(x,w,raw,y,oldx,oldw) in zip(lengths,arrays):
            if m:
                ref=(x[:,:k].double()@w[:,:k].double().T).to(torch.int32)
                assert torch.equal(y,ref),('full-range INT8 mismatch',lengths,n,k)
                mask=torch.ones_like(raw,dtype=torch.bool)
                view=torch.as_strided(mask[16:],(m,n),(n+4,1));view.fill_(False)
                assert bool((raw[mask]==123456789).all()),'Output guard or row padding overwritten'
            else:assert bool((raw==123456789).all())
            assert torch.equal(x,oldx) and torch.equal(w,oldw),'Input modified'
        records.append(dict(lengths=lengths,N=n,K=k,status='PASS',full_range_int8=True,nondefault_stream=True,padded_strides=True,guards=True))
    finally:lib.int8_destroy(ctx)
    print(json.dumps(records[-1]),flush=True)
(ROOT/'results/boundaries_public_v73.json').write_text(json.dumps(records,indent=2));print('BOUNDARIES PASS')
