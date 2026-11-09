"""Route25:borrowed scales,prepare/run streams,guards and queued destruction."""
from pathlib import Path
import ctypes as C,os,json,hashlib
import torch
from bindings import Weight,init,lib,ok,plan_info
from test_scaled_api import Group2
assert os.environ['HIP_VISIBLE_DEVICES']=='7'
torch.set_num_threads(4);torch.manual_seed(20261106)
records=[]
for m,n,k in [(1,4096,7168),(2,4352,7296)]:
 g=12;astride=k+16;bstride=k+32;cstride=n+8
 a=torch.randint(-128,128,(g,m,astride),device='cuda',dtype=torch.int8)
 b=torch.randint(-128,128,(g,n,bstride),device='cuda',dtype=torch.int8)
 a0,b0=a.clone(),b.clone()
 ref=torch.stack([(a[i,:,:k].double()@b[i,:,:k].double().T).int() for i in range(g)])
 weights=C.c_void_p();prep=torch.cuda.Stream();prep.wait_stream(torch.cuda.current_stream())
 wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=bstride) for i in range(g)])
 ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,prep.cuda_stream,C.byref(weights)))
 for typ,dtype in [(1,torch.bfloat16),(2,torch.float16)]:
  for outer in [False,True]:
   sa=torch.rand((g,m if outer else 1),device='cuda')*.01
   sb=torch.rand((g,n),device='cuda')*.01
   raw=torch.full((g,m*cstride+32),77,device='cuda',dtype=dtype);c=raw[:,16:-16].view(g,m,cstride)
   gd=(Group2*g)()
   for i in range(g):
    d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=c[i].data_ptr();d.m=m;d.a_row_stride=astride;d.c_row_stride=cstride;d.output_type=typ;d.scale_a=sa[i].data_ptr();d.scale_b=sb[i].data_ptr();d.scale_a_mode=int(outer);d.scale_b_mode=1
   plan=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan)));assert plan_info(plan).route==25
   assert lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights)==-7
   for epoch in range(3):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
     sa.mul_(.5)
     sb[:,0]=float('nan') if epoch==1 else 0.
     sb[:,1]=float('inf') if epoch==1 else -.001
     sb[:,2]=float('-inf') if epoch==1 else .002
    stream.synchronize();oldsa,oldsb=sa.clone(),sb.clone();stream.wait_stream(torch.cuda.current_stream())
    ok(lib.ck_tile_int8_nt_cached_run_v1(plan,stream.cuda_stream))
    if epoch==2:ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan));plan.value=None
    else:stream.synchronize()
    factor=sa.unsqueeze(-1)*sb.unsqueeze(1);expected=(ref.float()*factor).to(dtype);got=c[:,:,:n]
    assert ((got==expected)|(torch.isnan(got)&torch.isnan(expected))).all()
    assert (raw[:,:16]==77).all() and (raw[:,-16:]==77).all() and (c[:,:,n:]==77).all()
    assert torch.equal(a,a0) and torch.equal(b,b0)
    assert torch.equal(sa,oldsa) and ((sb==oldsb)|(torch.isnan(sb)&torch.isnan(oldsb))).all()
    records.append(dict(M=m,N=n,K=k,output=typ,outer_a=outer,epoch=epoch,status='PASS'))
 ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
path=Path(os.environ['CK_INT8_CACHED_BUILD'])/'libck_tile_int8_nt_cached.so'
report=dict(status='PASS',checks=len(records),library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records)
Path(os.environ['CK_INT8_K_LIFETIME_OUTPUT']).write_text(json.dumps(report,indent=2))
print('K_PARALLEL_LIFETIME_PASS',len(records))
