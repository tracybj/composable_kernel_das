"""GPU numerical gates for fused INT8 -> BF16/FP16 and unchanged INT32 APIs."""
from pathlib import Path
import csv,ctypes as C,json,os,hashlib,time,argparse,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
from bindings import Weight,Group,init,lib,ok,plan_info

class Group2(C.Structure):
    _fields_=[('struct_size',C.c_uint32),('abi_version',C.c_uint32),('a',C.c_void_p),('c',C.c_void_p),('m',C.c_int32),('a_row_stride',C.c_int32),('c_row_stride',C.c_int32),('output_type',C.c_uint32),('scale_a',C.c_void_p),('scale_b',C.c_void_p),('scale_a_mode',C.c_uint32),('scale_b_mode',C.c_uint32),('reserved',C.c_uint64*2)]
assert C.sizeof(Group2)==80
lib.ck_tile_int8_nt_cached_plan_create_v2.argtypes=[C.c_void_p,C.POINTER(Group2),C.c_int32,C.POINTER(C.c_void_p)]
E=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--blas-reference',type=Path);args=ap.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261104)
    source=list(csv.DictReader((E/'docs/current_refactor_v121_1178.csv').open()))
    routes={}
    for r in source:
        key=(int(r['route']),int(r['legacy_instance']) if int(r['route'])==0 else 0)
        if key not in routes or int(r['G'])*int(r['M'])*int(r['N'])*int(r['K'])<int(routes[key]['G'])*int(routes[key]['M'])*int(routes[key]['N'])*int(routes[key]['K']):routes[key]=r
    cases=[]
    for key,r in sorted(routes.items()):
        g,m,n,k=[int(r[x]) for x in ['G','M','N','K']]
        scaled=key
        if m>=2048 and m%256==0 and 3584<=n<=8192 and n%256==0 and n%384!=0 and 1024<=k<16384 and k%128==0:
            scaled=(23 if k>=4096 or n%384==128 else 22,0)
        cases.append((f'route_{key}',g,[m]*g,n,k,False,{0:key,1:scaled,2:scaled}))
    cases += [('tail_padded_ragged',4,[0,1,17,33],260,272,True,None),('special_values',2,[16]*2,256,512,False,None),('rounding_ties',2,[16]*2,256,512,False,None)]
    cases += [(f'supplement_M{m}',64,[m]*64,256,512,False,None) for m in [1,2,4,8,16,32,64,128,256,2048,4096,8192,16384]]
    # Exercise the new scaled policy through the production API,including NULL
    # scales and per-token fallback. INT32 keeps the original route in each case.
    cases += [('scaled_policy_low_n_m256',2,[256]*2,256,512,False,{0:(3,0),1:(8,0),2:(8,0)}),
              ('scaled_policy_low_n_m2048',2,[2048]*2,256,512,False,{0:(3,0),1:(6,0),2:(6,0)}),
              ('scaled_policy_k768',2,[2048]*2,2048,768,False,{0:(4,0),1:(6,0),2:(6,0)})]
    cases += [('scaled_policy_tail_cl4',2,[2048]*2,4096,1536,False,{0:(5,0),1:(22,0),2:(22,0)}),
              ('scaled_policy_tail_cl8_n',2,[2048]*2,5120,2304,True,{0:(5,0),1:(23,0),2:(23,0)}),
              ('scaled_policy_tail_cl8_k',2,[2048]*2,4096,4096,False,{0:(5,0),1:(23,0),2:(23,0)})]
    records=[];invalid_checks=0;covered=set()
    for name,g,ms,n,k,padded,expected in cases:
        astride=k+16 if padded else k;bstride=k+32 if padded else k;cstride=n+8 if padded else n
        maxm=max(ms);a=torch.randint(-128,128,(g,maxm,astride),device='cuda',dtype=torch.int8);b=torch.randint(-128,128,(g,n,bstride),device='cuda',dtype=torch.int8)
        if name=='rounding_ties':a.fill_(1);a[1].fill_(-1);b.fill_(1)
        wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=bstride) for i in range(g)])
        weights=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)));torch.cuda.synchronize()
        # FP64 reference preserves every integer dot product across full INT8 range.
        refs=[]
        for i,m in enumerate(ms):refs.append((a[i,:m,:k].double()@b[i,:,:k].double().T).to(torch.int32))
        for mode in ['none','scalar','channel','row_channel']+(['special'] if name=='special_values' else [])+(['ties'] if name=='rounding_ties' else []):
            sa=None if mode=='none' else torch.rand((g,maxm if mode in ['row_channel','special'] else 1),device='cuda')*.003+.001
            sb=None if mode=='none' else torch.rand((g,n if mode in ['channel','row_channel','special','ties'] else 1),device='cuda')*.003+.001
            if mode=='ties':
                sa.fill_(1/k)
                sb[:]=torch.tensor([1+2**-8,1+3*2**-8,1+2**-11,1+3*2**-11],device='cuda').repeat(n//4)
            if mode=='special':
                sa[:,:8]=torch.tensor([0.,-0.,1e-35,1e30,float('inf'),float('-inf'),float('nan'),-1.],device='cuda')
                sb[:,0:4]=torch.tensor([0.,-.5,1.,1e-35],device='cuda')
            for output,dtype in [(1,torch.bfloat16),(2,torch.float16)]+([(0,torch.int32)] if mode=='none' else []):
                span=maxm*cstride;capacity=(span+7)//8*8
                raw=torch.full((g,capacity+32),77,device='cuda',dtype=dtype);out=raw[:,16:16+span].view(g,maxm,cstride)
                gd=(Group2*g)()
                for i,m in enumerate(ms):
                    d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=out[i].data_ptr();d.m=m;d.a_row_stride=astride;d.c_row_stride=cstride;d.output_type=output
                    d.scale_a=sa[i].data_ptr() if sa is not None else None;d.scale_b=sb[i].data_ptr() if sb is not None else None
                    d.scale_a_mode=int(mode in ['row_channel','special']);d.scale_b_mode=int(mode in ['channel','row_channel','special','ties'])
                plan=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan)));pi=plan_info(plan)
                if expected is not None:
                    want=expected[output] if isinstance(expected,dict) else expected
                    assert (pi.route,pi.legacy_instance if pi.route==0 else 0)==want,(name,output,pi.route,want)
                covered.add((pi.route,pi.legacy_instance if pi.route==0 else 0))
                ok(lib.ck_tile_int8_nt_cached_run_v1(plan,None));torch.cuda.synchronize()
                bad=0
                for i,m in enumerate(ms):
                    ref=refs[i] if output==0 else refs[i].float()
                    if sa is not None:
                        scalea=sa[i,:m,None] if gd[i].scale_a_mode else sa[i,0]
                        scaleb=sb[i,None,:] if gd[i].scale_b_mode else sb[i,0]
                        ref=ref*(scalea*scaleb)
                    ref=ref.to(dtype);got=out[i,:m,:n]
                    equal=got==ref
                    if output!=0:equal=equal|(torch.isnan(got)&torch.isnan(ref))
                    count=int((~equal).sum());bad+=count
                    if count:
                        positions=(~equal).nonzero()[:8]
                        detail=[]
                        for rr,cc in positions.tolist():
                            va=float(sa[i,rr if gd[i].scale_a_mode else 0]) if sa is not None else 1.
                            vb=float(sb[i,cc if gd[i].scale_b_mode else 0]) if sb is not None else 1.
                            dot=refs[i][rr,cc].float()
                            detail.append(dict(row=rr,col=cc,dot=int(refs[i][rr,cc]),sa=va,sb=vb,got=float(got[rr,cc]),expected=float(ref[rr,cc]),sequential_a_b=float(((dot*va)*vb).to(dtype)),sequential_b_a=float(((dot*vb)*va).to(dtype)),grouped=float((dot*(torch.tensor(va,device='cuda')*vb)).to(dtype))))
                        print('MISMATCH_DETAIL',i,json.dumps(detail),flush=True)
                    assert bool((out[i,m:,:]==77).all()) and bool((out[i,:m,n:]==77).all()),'row/stride guard'
                assert bad==0,(name,output,mode,bad)
                assert bool((raw[:,:16]==77).all()) and bool((raw[:,16+span:]==77).all()),'allocation guards'
                # The old v1 entry still writes exact INT32 results.
                if output==0:
                    old=(Group*g)(*[init(Group,a=a[i].data_ptr(),c=out[i].data_ptr(),m=ms[i],a_row_stride=astride,c_row_stride=cstride) for i in range(g)])
                    other=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v1(weights,old,g,C.byref(other)));ok(lib.ck_tile_int8_nt_cached_run_v1(other,None));torch.cuda.synchronize()
                    assert all(torch.equal(out[i,:m,:n],refs[i]) for i,m in enumerate(ms))
                    ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(other))
                ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                rec=dict(case=name,output=str(dtype),scale_mode=mode,route=pi.route,legacy_instance=pi.legacy_instance,status='PASS',bad_elements=bad)
                records.append(rec);print(json.dumps(rec),flush=True)
                if name=='special_values' and mode=='none' and output==0:
                    for field,value in [('abi_version',1),('struct_size',79),('output_type',3),('output_type',1),('scale_a_mode',2),('scale_a',a.data_ptr()),('reserved',1)]:
                        d=gd[0];previous=d.reserved[0] if field=='reserved' else getattr(d,field)
                        if field=='reserved':d.reserved[0]=value
                        else:setattr(d,field,value)
                        invalid=C.c_void_p();assert lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(invalid))!=0 and not invalid.value
                        if field=='reserved':d.reserved[0]=previous
                        else:setattr(d,field,previous)
                        invalid_checks+=1
                del raw,out,gd
            del sa,sb
        ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights));del a,b,refs;torch.cuda.empty_cache()
    native=native_compare(args.blas_reference) if args.blas_reference else []
    api_path=Path(os.environ['CK_INT8_CACHED_BUILD'])/'libck_tile_int8_nt_cached.so'
    report=dict(status='PASS',cases=len(cases),checks=len(records),invalid_checks=invalid_checks,covered_routes=sorted(covered),full_range_int8=True,reference='FP64 exact dot -> INT32 -> FP32 dot*(scaleA*scaleB) -> selected16bit RNE',library_sha256=hashlib.sha256(api_path.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_blas_checks=native,records=records)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print('SCALED_API_PASS',len(records),flush=True)
def native_compare(path):
    """Compare actual native BLASLt and fused CK outputs,without a tolerance."""
    blas=C.CDLL(str(path))
    class D(C.Structure):
        _fields_=[('a',C.c_void_p),('b',C.c_void_p),('c',C.c_void_p),('sa',C.c_void_p),('sb',C.c_void_p),('m',C.c_int),('n',C.c_int),('k',C.c_int),('as_',C.c_int),('bs',C.c_int),('cs',C.c_int)]
    blas.blas_scale_reference.argtypes=[C.c_int,C.POINTER(D),C.c_int]
    blas.blas_scale_reference_error.restype=C.c_char_p
    blas.blas_scale_reference_loaded_path.restype=C.c_char_p
    loaded=Path(blas.blas_scale_reference_loaded_path().decode()).resolve()
    if os.environ.get('CK_INT8_BLAS_LIBRARY'):assert loaded==Path(os.environ['CK_INT8_BLAS_LIBRARY']).resolve()
    rows=[]
    for output,dtype in [(1,torch.bfloat16),(2,torch.float16)]:
        for m,n,k,g in [(16,256,512,2),(128,512,1024,4),(256,2048,512,8)]:
            a=torch.randint(-128,128,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-128,128,(g,n,k),device='cuda',dtype=torch.int8)
            ck=torch.zeros(g,m,n,device='cuda',dtype=dtype);ref=torch.zeros_like(ck)
            sa=torch.tensor([.031415927*(i+1) for i in range(g)],device='cuda');sb=torch.tensor([.027182818/(i+1) for i in range(g)],device='cuda')
            wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=k) for i in range(g)])
            gd=(Group2*g)()
            for i in range(g):
                d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=ck[i].data_ptr();d.m=m;d.a_row_stride=k;d.c_row_stride=n;d.output_type=output;d.scale_a=sa[i:].data_ptr();d.scale_b=sb[i:].data_ptr()
            weights,plan=C.c_void_p(),C.c_void_p();ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)));ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan)))
            ok(lib.ck_tile_int8_nt_cached_run_v1(plan,None));torch.cuda.synchronize()
            ds=(D*g)(*[D(a[i].data_ptr(),b[i].data_ptr(),ref[i].data_ptr(),sa[i:].data_ptr(),sb[i:].data_ptr(),m,n,k,k,k,n) for i in range(g)])
            assert blas.blas_scale_reference(g,ds,output)==0,blas.blas_scale_reference_error().decode()
            bad=int((ck.view(torch.int16)!=ref.view(torch.int16)).sum());assert bad==0,('NATIVE_BLAS_MISMATCH',output,m,n,k,g,bad)
            row=dict(output=str(dtype),M=m,N=n,K=k,G=g,bitwise_mismatches=bad,status='PASS',blas_library=str(loaded));rows.append(row);print('NATIVE_BLAS',json.dumps(row),flush=True)
            ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan));ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
    return rows

if __name__=='__main__':main()
