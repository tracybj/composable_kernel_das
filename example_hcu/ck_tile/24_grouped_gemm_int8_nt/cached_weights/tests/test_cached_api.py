"""New INT8-only API:full-range FP64,packed bytes,streams,lifetimes and contracts."""
from pathlib import Path
import ctypes as C,json,os,resource,hashlib,gc
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
from bindings import *
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4)
cases=[([17,17],260,528),([32]*8,260,1232),([33,33],260,1232),([64]*8,260,4112),([1,17,0],1236,1232),([0,0],128,512),([33,0,257],300,528),([128,256],260,256),([128,128],256,512),([256,256],256,2048),([128],256,2048),([257,257],260,2064),([129,129],260,528),([4096],256,2048),([4096],256,4096),([4097],260,1232),([16,16],256,4096),([1],260,256),([2,2],260,768),([3],260,1232),([8]*17,260,2064),([16]*17,260,528),([1],260,4112),([16]*8,260,6160),([3]*8,260,4096),([256],256,384),([512],256,640),([2048],256,768),([2304],256,768),([2304],256,896),([256],512,4096),([256],512,6144),([512],256,6144),([512],256,7168),([512],256,1536),([512],256,3072),([256],512,7168),([512],256,768),([512],256,640),([768],512,6144),([256],768,6144),([2048],256,1536),([2048],512,3072),([4096],1280,2048),([8192],1024,2048),([4096],1024,2048)]
results=[]
for index,(lengths,n,k) in enumerate(cases):
 torch.manual_seed(20261029+index);g=len(lengths)
 sa=k+16;sb=k+32
 if index in [30,41]:sa=16384;sb=16400
 arrays=[];weights=[];groups=[]
 for m in lengths:
  x=torch.randint(-128,128,(m,sa),device='cuda',dtype=torch.int8);w=torch.randint(-128,128,(n,sb),device='cuda',dtype=torch.int8)
  raw=torch.full((m*(n+4)+32,),123456789,device='cuda',dtype=torch.int32);y=torch.as_strided(raw[16:],(m,n),(n+4,1))
  arrays.append((x,w,raw,y,x.clone(),w.clone()));weights.append(init(Weight,data=w.data_ptr(),n=n,k=k,row_stride=sb));groups.append(init(Group,a=x.data_ptr(),c=y.data_ptr(),m=m,a_row_stride=sa,c_row_stride=n+4))
 wd=(Weight*g)(*weights);gd=(Group*g)(*groups);handle=C.c_void_p();plan=C.c_void_p();bench=None
 load_stream=torch.cuda.Stream();load_stream.wait_stream(torch.cuda.current_stream())
 ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,load_stream.cuda_stream,C.byref(handle)))
 try:
  wi=weight_info(handle);assert (wi.group_count,wi.n,wi.k)==(g,n,k)
  ok(lib.ck_tile_int8_nt_cached_plan_create_v1(handle,gd,g,C.byref(plan)));pi=plan_info(plan);assert pi.max_m==max(lengths)
  assert lib.ck_tile_int8_nt_cached_weights_destroy_v1(handle)==-7,'Live plan must retain weights'
  compute=torch.cuda.Stream();bench=lib.cached_i8_bench_create(plan);assert bench,lib.cached_i8_bench_error().decode()
  ms=float(lib.cached_i8_bench_run(bench,0,2,compute.cuda_stream));assert ms>0,lib.cached_i8_bench_error().decode();compute.synchronize()
  for m,(x,w,raw,y,oldx,oldw) in zip(lengths,arrays):
   if m:
    ref=(x[:,:k].double()@w[:,:k].double().T).to(torch.int32);assert torch.equal(y,ref),(index,lengths,n,k,pi.route)
    mask=torch.ones_like(raw,dtype=torch.bool);torch.as_strided(mask[16:],(m,n),(n+4,1)).fill_(False);assert bool((raw[mask]==123456789).all())
   else:assert bool((raw==123456789).all())
   assert torch.equal(x,oldx) and torch.equal(w,oldw)
  byte_check=False
  if wi.format==1 and wi.packed_bytes<=4*1024**2:
   # hipMemcpyDeviceToHost=2 in the HIP runtime ABI;the handle's dependency
   # lookup resolves the same runtime used by this extension. Compute waited
   # on preparation,so the read-only packed buffer is complete here.
   lib.hipMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int];lib.hipMemcpy.restype=C.c_int
   data=(C.c_ubyte*wi.packed_bytes)();assert lib.hipMemcpy(data,wi.packed_device_data,wi.packed_bytes,2)==0
   expected=torch.stack([z[1][:,:k].cpu() for z in arrays]).reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous()
   assert bytes(data)==expected.numpy().tobytes(),('Packed bytes mismatch',index);byte_check=True
  # Strict version/reserved/group-count gates must not mutate the live plan.
  bad=C.c_void_p(123);gd[0].abi_version=2;assert lib.ck_tile_int8_nt_cached_plan_create_v1(handle,gd,g,C.byref(bad))==-2 and not bad.value;gd[0].abi_version=1
  bad=C.c_void_p(123);gd[0].reserved[0]=1;assert lib.ck_tile_int8_nt_cached_plan_create_v1(handle,gd,g,C.byref(bad))==-1 and not bad.value;gd[0].reserved[0]=0
  bad=C.c_void_p(123);assert lib.ck_tile_int8_nt_cached_plan_create_v1(handle,gd,g+1,C.byref(bad))==-1 and not bad.value
  results.append(dict(index=index,lengths=lengths,N=n,K=k,route=pi.route,legacy_instance=pi.legacy_instance,status='PASS',full_range_int8=True,fp64_reference=True,nondefault_prepare_and_compute_streams=True,packed_bytes_checked=byte_check,inputs_unchanged=True,output_guards=True,weight_lifetime_and_contracts=True))
  print(json.dumps(results[-1]),flush=True)
 finally:
  if bench:lib.cached_i8_bench_destroy(bench)
  if plan.value:ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
  ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(handle))
 del arrays,weights,groups,wd,gd,x,w,raw,y,oldx,oldw;gc.collect();torch.cuda.empty_cache()
assert {0,1,3,4,5}<={x['route'] for x in results},'Exercise every production route in this policy'
report=dict(status='PASS',checks=results,library_sha256=hashlib.sha256((E/'build/cached_v97/libck_tile_int8_nt_cached.so').read_bytes()).hexdigest(),routes=sorted({x['route'] for x in results}),C_ABI_sizes=[C.sizeof(t) for t in [Weight,Group,WeightInfo,PlanInfo]])
(E/'results/cached_api_v97_boundaries.json').write_text(json.dumps(report,indent=2)+'\n');print('CACHED_API_BOUNDARIES_PASS',len(results))
