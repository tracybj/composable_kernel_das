"""Full-output native epilogue gates: special scales, live updates and wide tails."""
from pathlib import Path
import argparse,json,os,ctypes as C,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
from bindings import Weight,init,lib,ok,plan_info
from test_scaled_api import Group2

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert os.environ.get('HIP_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(4);torch.manual_seed(20261104)
    records=[]
    for m,n,k,g,route in [(128,256,512,2,7),(256,512,512,2,3),(2048,1792,1536,2,12)]:
        a=torch.randint(-128,128,(g,m,k),device='cuda',dtype=torch.int8)
        b=torch.randint(-128,128,(g,n,k),device='cuda',dtype=torch.int8)
        refs=[(a[i].double()@b[i].double().T).to(torch.int32).float() for i in range(g)]
        sa=torch.tensor([.003,-.005],device='cuda');sb=torch.rand(g,n,device='cuda')*.01
        wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=k) for i in range(g)])
        weights=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)))
        try:
            for output,dtype in [(1,torch.bfloat16),(2,torch.float16)]:
                raw=torch.full((g,m*n+32),77,device='cuda',dtype=dtype);out=raw[:,16:-16].view(g,m,n)
                gd=(Group2*g)()
                for i in range(g):
                    d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=out[i].data_ptr();d.m=m;d.a_row_stride=k;d.c_row_stride=n;d.output_type=output
                    d.scale_a=sa[i:].data_ptr();d.scale_b=sb[i].data_ptr();d.scale_b_mode=1
                plan=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan)))
                try:
                    pi=plan_info(plan);assert pi.route==route,(m,n,k,g,pi.route,route)
                    for step in range(3):
                        if step==0:
                            sa.copy_(torch.tensor([.003,-.005],device='cuda'));sb.uniform_(-.01,.01)
                            sb[:,:8]=torch.tensor([0.,-0.,float('inf'),float('-inf'),float('nan'),1e-35,1e30,-.5],device='cuda')
                        elif step==1:sa.mul_(2);sb[:,8:].mul_(-.5)
                        else:sa.zero_();sb.fill_(float('inf'))
                        ok(lib.ck_tile_int8_nt_cached_run_v1(plan,None));torch.cuda.synchronize()
                        bad=0
                        for i in range(g):
                            ref=(refs[i]*(sa[i]*sb[i])[None,:]).to(dtype)
                            equal=(out[i].view(torch.int16)==ref.view(torch.int16))|(torch.isnan(out[i])&torch.isnan(ref))
                            bad+=int((~equal).sum())
                        assert bad==0,(m,n,k,g,output,step,bad)
                        assert bool((raw[:,:16]==77).all()&(raw[:,-16:]==77).all())
                        rec=dict(M=m,N=n,K=k,G=g,route=route,output=str(dtype),step=step,status='PASS',bad_elements=bad,full_output=True)
                        records.append(rec);print(json.dumps(rec),flush=True)
                finally:ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                del raw,out,gd
        finally:ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
        del a,b,refs,sa,sb,wd;torch.cuda.empty_cache()
    args.output.write_text(json.dumps(dict(status='PASS',checks=records),indent=2)+'\n')
    print('SCALED_EPILOGUE_PASS',len(records),flush=True)

if __name__=='__main__':main()
