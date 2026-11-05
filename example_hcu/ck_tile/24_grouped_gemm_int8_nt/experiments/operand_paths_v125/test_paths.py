"""Full-range gates for experimental operand paths and M128 wide WASP."""
from pathlib import Path
import argparse,ctypes as C,hashlib,json,os,resource,sys
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
E=Path(__file__).resolve().parents[2];sys.path.insert(0,str(E/'cached_weights/tests'))
import torch
from bindings import Weight,init,lib,ok,plan_info
from test_scaled_api import Group2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--shuffle-only',action='store_true');ap.add_argument('--deferred-only',action='store_true');ap.add_argument('--next-round-only',action='store_true');ap.add_argument('--packed-input-only',action='store_true');ap.add_argument('--fullnk-grid-only',action='store_true');ap.add_argument('--address32-only',action='store_true');ap.add_argument('--tail-mask-only',action='store_true');args=ap.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
    build=Path(os.environ['CK_INT8_CACHED_BUILD']);identity=json.loads((build/'exported_api_symbols.json').read_text());assert identity['operand_paths']
    torch.set_num_threads(4);torch.manual_seed(20261104)
    records=[];rejected=0
    cases=[(1,256,512,False),(17,260,1232,True),(129,512,528,True),(128,256,8192,False),(256,1792,1536,True),(256,4096,7168,False)]
    candidates=[(0,i) for i in range(9013,9022)] if args.shuffle_only else [(0,i) for i in range(9001,9022)]+[(r,None) for r in range(13,18)]
    if args.deferred_only:candidates=[(0,i) for i in [9022,9023,9024,9031,9032,9034,9035,9036]]
    if args.next_round_only:candidates=[(0,i) for i in range(9041,9050)]+[(r,None) for r in range(18,22)]
    if args.packed_input_only:candidates=[(0,i) for i in range(9051,9058)]+[(r,None) for r in range(32,36)]
    if args.fullnk_grid_only:
        candidates=[(r,None) for r in [*range(32,40),*range(48,56)]]
        cases+=[(17,512,768,True)] # legal pack with K512 tail:checked paths must work;FullNK512 rejects.
    if args.address32_only:
        candidates=[(r,None) for r in range(36,44)]
        cases+=[(17,512,768,True)]
    if args.tail_mask_only:
        candidates=[(r,None) for r in [12,48,49,50,51,*range(56,61)]]
        cases += [(256,256,512,False),(512,512,640,True),
                  (256,5120,2304,False),(256,6144,2048,False),(256,7168,2048,True)]
    for m,n,k,padded in cases:
        g=2;astride=k+16 if padded else k;bstride=k+32 if padded else k;cstride=n+4
        a=torch.randint(-128,128,(g,m,astride),device='cuda',dtype=torch.int8);b=torch.randint(-128,128,(g,n,bstride),device='cuda',dtype=torch.int8)
        olda,oldb=a.clone(),b.clone()
        refs=[(a[i,:,:k].double()@b[i,:,:k].double().T).to(torch.int32) for i in range(g)]
        wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=bstride) for i in range(g)])
        weights=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)));torch.cuda.synchronize()
        try:
            for route,legacy in candidates:
                os.environ['CK_INT8_TUNE_ROUTE']=str(route);os.environ.pop('CK_INT8_TUNE_LEGACY',None)
                if legacy is not None:os.environ['CK_INT8_TUNE_LEGACY']=str(legacy)
                for mode in ['none','channel','row_channel','special']:
                    sa=None if mode=='none' else torch.rand((g,m if mode=='row_channel' else 1),device='cuda')*.003+.001
                    sb=None if mode=='none' else torch.rand((g,n),device='cuda')*.003+.001
                    if mode=='special':
                        sa[:,0]=torch.tensor([0.,-.003],device='cuda')
                        sb[:,:8]=torch.tensor([float('inf'),float('-inf'),float('nan'),-0.,0.,1e-35,1e30,-.5],device='cuda')
                    outputs=[(1,torch.bfloat16),(2,torch.float16)]+([(0,torch.int32)] if mode=='none' else [])
                    for output,dtype in outputs:
                        span=m*cstride;cap=(span+7)//8*8
                        raw=torch.full((g,cap+32),77,device='cuda',dtype=dtype);out=raw[:,16:16+span].view(g,m,cstride)
                        gd=(Group2*g)()
                        for i in range(g):
                            d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=out[i].data_ptr();d.m=m;d.a_row_stride=astride;d.c_row_stride=cstride;d.output_type=output
                            d.scale_a=sa[i].data_ptr() if sa is not None else None;d.scale_b=sb[i].data_ptr() if sb is not None else None;d.scale_a_mode=int(mode=='row_channel');d.scale_b_mode=int(mode!='none')
                        plan=C.c_void_p();status=lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan))
                        if status:
                            # Native-domain checks must reject unsupported WASP
                            # shapes before launch;all cooperative paths are masked.
                            assert route>=13,(route,legacy,m,n,k,status,lib.ck_tile_int8_nt_cached_last_error_v1().decode())
                            rejected+=1;del raw,out,gd;continue
                        try:
                            pi=plan_info(plan);assert pi.route==route and (legacy is None or pi.legacy_instance==legacy)
                            ok(lib.ck_tile_int8_nt_cached_run_v1(plan,None));torch.cuda.synchronize()
                            bad=0
                            for i in range(g):
                                ref=refs[i] if output==0 else refs[i].float()
                                if mode!='none':
                                    aa=sa[i,:m,None] if mode=='row_channel' else sa[i,0]
                                    ref=ref*(aa*sb[i,None,:])
                                ref=ref.to(dtype);got=out[i,:,:n]
                                equal=got==ref if output==0 else (got.view(torch.int16)==ref.view(torch.int16))|(torch.isnan(got)&torch.isnan(ref))
                                bad+=int((~equal).sum())
                            assert bad==0,('NUMERICAL_FAIL',m,n,k,route,legacy,output,mode,bad)
                            assert bool((raw[:,:16]==77).all()&(raw[:,16+span:]==77).all()&(out[:,:,n:]==77).all()),'OUTPUT_GUARD'
                            rec=dict(M=m,N=n,K=k,G=g,padded=padded,route=route,legacy_instance=legacy,output=str(dtype),scale_mode=mode,status='PASS',bad_elements=bad,full_range_int8=True,full_output=True)
                            records.append(rec);print(json.dumps(rec),flush=True)
                        finally:ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                        del raw,out,gd
                    del sa,sb
            assert torch.equal(a,olda) and torch.equal(b,oldb),'INPUT_CHANGED'
        finally:ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
        del a,b,olda,oldb,refs,wd;torch.cuda.empty_cache()
    report=dict(status='PASS',library=identity,checks=len(records),rejected_before_launch=rejected,records=records,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2)+'\n');print('OPERAND_PATH_GATES_PASS',len(records),flush=True)

if __name__=='__main__':main()
