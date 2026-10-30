"""Explicit weights reuse/refresh and plan-stream lifetime; no pointer cache."""
from pathlib import Path
import ctypes as C,json,resource,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
from bindings import *
torch.set_num_threads(4);torch.manual_seed(20261029)
g,n,k=2,256,2048
weights=[torch.randint(-128,128,(n,k),device='cuda',dtype=torch.int8) for _ in range(g)]
original_ptrs=[x.data_ptr() for x in weights];checks=[]
for generation in range(2):
 if generation:
  # Every prior handle/plan is destroyed before modifying the same addresses.
  for w in weights:w.add_(1)
 assert [x.data_ptr() for x in weights]==original_ptrs
 wd=(Weight*g)(*[init(Weight,data=w.data_ptr(),n=n,k=k,row_stride=k) for w in weights])
 h=C.c_void_p();prep=torch.cuda.Stream();prep.wait_stream(torch.cuda.current_stream())
 ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,prep.cuda_stream,C.byref(h)))
 wi=weight_info(h);assert wi.format==1;packed_ptr=wi.packed_device_data
 plans=[];storage=[]
 try:
  for m in [64,128,256,512,2048]:
   a=[torch.randint(-128,128,(m,k),device='cuda',dtype=torch.int8) for _ in range(g)]
   c=[torch.empty((m,n),device='cuda',dtype=torch.int32) for _ in range(g)]
   gd=(Group*g)(*[init(Group,a=x.data_ptr(),c=y.data_ptr(),m=m,a_row_stride=k,c_row_stride=n) for x,y in zip(a,c)])
   p=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v1(h,gd,g,C.byref(p)))
   plans.append(p);storage.append((a,c,gd,plan_info(p).route))
  assert lib.ck_tile_int8_nt_cached_weights_destroy_v1(h)==-7
  # Both plans reuse one immutable packed buffer; each is exercised on two streams.
  for p,(a,c,gd,route) in zip(plans,storage):
   for iteration in range(2):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):ok(lib.ck_tile_int8_nt_cached_run_v1(p,stream.cuda_stream))
    stream.synchronize()
    for x,y,w in zip(a,c,weights):assert torch.equal(y,(x.double()@w.double().T).to(torch.int32))
    assert weight_info(h).packed_device_data==packed_ptr
    checks.append(dict(generation=generation,M=a[0].shape[0],route=route,stream_iteration=iteration,exact_fp64=True,shared_weights=True,status='PASS'))
   # Destroy directly after another asynchronous run: destruction must await it.
   stream=torch.cuda.Stream();ok(lib.ck_tile_int8_nt_cached_run_v1(p,stream.cuda_stream))
   ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(p));p.value=None
   for x,y,w in zip(a,c,weights):assert torch.equal(y,(x.double()@w.double().T).to(torch.int32))
 finally:
  for p in plans:
   if p.value:ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(p))
  ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(h))
report=dict(status='PASS',checks=checks,same_source_addresses_refreshed=True,plans_destroy_wait_for_queued_work=True,library_sha256=hashlib.sha256((E/'build/cached_v106/libck_tile_int8_nt_cached.so').read_bytes()).hexdigest())
(E/'results/cached_api_v106_lifetime.json').write_text(json.dumps(report,indent=2)+'\n');print('CACHED_API_LIFETIME_PASS',len(checks))
