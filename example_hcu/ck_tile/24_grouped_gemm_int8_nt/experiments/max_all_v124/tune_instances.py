"""Screen existing instances;case IDs identify fixtures,never runtime dispatch."""
from pathlib import Path
import argparse,csv,ctypes as C,datetime,hashlib,json,os,resource,statistics,sys,traceback
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
E=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(E/'cached_weights/tests'))

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--suite',choices=['low_n','prefill','decode'],default='low_n')
    ap.add_argument('--warmup',type=int,default=3);ap.add_argument('--repeat',type=int,default=10);ap.add_argument('--samples',type=int,default=2)
    ap.add_argument('--validation',choices=['full','sample'],default='full');args=ap.parse_args()
    assert args.repeat>0 and args.samples>0
    assert all(os.environ.get(x)=='7' for x in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','DTK_VISIBLE_DEVICES'])
    import torch
    from bindings import Weight,init,lib,ok,plan_info,weight_info
    from test_scaled_api import Group2
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    prop=torch.cuda.get_device_properties(0);assert prop.gcnArchName.startswith('gfx936')
    lib.hipDeviceGetPCIBusId.argtypes=[C.c_void_p,C.c_int,C.c_int];lib.hipDeviceGetPCIBusId.restype=C.c_int
    bus=C.create_string_buffer(64);assert lib.hipDeviceGetPCIBusId(bus,64,0)==0
    assert bus.value.decode().lower()==Path('/sys/class/drm/renderD135/device').resolve().name.lower()
    targets=list(csv.DictReader((E/'docs/targets_max_all_1191.csv').open()))
    targets=[dict(**{k:int(r[k]) for k in ['ID','M','N','K','G']},phase=r['phase'],target_TOPS=float(r['target_TOPS'])) for r in targets]
    if args.suite=='low_n':
        cases=[r for r in targets if r['N']==256 and r['K']==512 and r['G']==64]
        cases += [dict(ID=f'extra_M{m}',M=m,N=256,K=512,G=64,phase='synthetic',target_TOPS=None) for m in [512,1024]]
    else:
        baseline=list(csv.DictReader((E/'docs/max_all_v123_baseline.csv').open()))
        # One representative per dimension/route family,then largest absolute misses.
        missed=[r for r in baseline if r['phase']==args.suite and float(r['BF16_R'])<1]
        selected={}
        for r in sorted(missed,key=lambda r:float(r['BF16_R'])):
            key=(r['route'],r['N'],r['K'])
            if key not in selected:selected[key]=r
        ids={int(r['ID']) for r in list(selected.values())[:16]}
        cases=[r for r in targets if r['ID'] in ids]
    candidates=[('default',None,None)]+[(f'route_{r}',r,None) for r in [1,3,5,6,7,8,9,10,11,12]]
    candidates += [(f'cooperative_{i}',0,i) for i in [6005,6006,6035,6041,6042,6061,7501,7603]]
    if args.suite=='decode':candidates += [(f'cooperative_{i}',0,i) for i in [6401,6403,6404,6505,7502,7504,7604]]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    assert not (out/'records.jsonl').exists(),'Existing run;inspect before restarting'
    build=Path(os.environ['CK_INT8_CACHED_BUILD']);identity=json.loads((build/'exported_api_symbols.json').read_text())
    assert identity['experimental_tuning']
    atomic(out/'manifest.json',dict(suite=args.suite,cases=cases,candidates=candidates,validation=args.validation,warmup=args.warmup,repeat=args.repeat,samples=args.samples,library=identity,gpu=7,pci=bus.value.decode(),input_range=[-8,8],scale='FP32 scalar A + FP32 channel B',target_sha256=hashlib.sha256((E/'docs/targets_max_all_1191.csv').read_bytes()).hexdigest()))
    records=[];best=[]
    try:
        with (out/'records.jsonl').open('w') as stream:
            for case_index,case in enumerate(cases):
                ID,m,n,k,g=[case[x] for x in ['ID','M','N','K','G']]
                atomic(out/'active.json',dict(**case,stage='allocate'))
                torch.manual_seed(20261104+(ID if isinstance(ID,int) else 100000+case_index))
                a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
                sa=torch.rand(g,device='cuda')*.008+.002;sb=torch.rand(g,n,device='cuda')*.008+.002
                span=m*n;capacity=(span+7)//8*8
                raw={kind:torch.full((g,capacity+32),117,device='cuda',dtype=dtype) for kind,dtype in [('BF16',torch.bfloat16),('FP16',torch.float16)]}
                output={kind:v[:,16:16+span].view(g,m,n) for kind,v in raw.items()}
                indices=torch.arange(m,device='cuda') if args.validation=='full' else torch.linspace(0,m-1,min(m,8),device='cuda').long()
                refs={'BF16':[],'FP16':[]}
                for i in range(g):
                    scaled=(a[i,indices].float()@b[i].float().T)*(sa[i]*sb[i])[None,:]
                    refs['BF16'].append(scaled.to(torch.bfloat16));refs['FP16'].append(scaled.to(torch.float16))
                del scaled
                sampled=[]
                for x in [a,b,sa,sb]:
                    ix=torch.linspace(0,x.numel()-1,min(x.numel(),128),device='cuda').long();sampled.append((ix,x.reshape(-1)[ix].clone()))
                wd=(Weight*g)(*[init(Weight,data=b[i].data_ptr(),n=n,k=k,row_stride=k) for i in range(g)])
                weights=C.c_void_p();ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd,g,None,C.byref(weights)));torch.cuda.synchronize()
                wi=weight_info(weights);plans={};benches={};metadata={};timings={}
                try:
                    for label,route,legacy in candidates:
                        for key in ['CK_INT8_TUNE_ROUTE','CK_INT8_TUNE_LEGACY']:os.environ.pop(key,None)
                        if route is not None:os.environ['CK_INT8_TUNE_ROUTE']=str(route)
                        if legacy is not None:os.environ['CK_INT8_TUNE_LEGACY']=str(legacy)
                        for code,kind in [(1,'BF16'),(2,'FP16')]:
                            gd=(Group2*g)()
                            for i in range(g):
                                d=gd[i];d.struct_size=C.sizeof(d);d.abi_version=2;d.a=a[i].data_ptr();d.c=output[kind][i].data_ptr();d.m=m;d.a_row_stride=k;d.c_row_stride=n;d.output_type=code;d.scale_a=sa[i:].data_ptr();d.scale_b=sb[i].data_ptr();d.scale_b_mode=1
                            plan=C.c_void_p();status=lib.ck_tile_int8_nt_cached_plan_create_v2(weights,gd,g,C.byref(plan))
                            if status:
                                stream.write(json.dumps(dict(**case,candidate=label,output=kind,status='UNSUPPORTED',error=lib.ck_tile_int8_nt_cached_last_error_v1().decode()))+'\n');continue
                            key=(label,kind);plans[key]=plan;pi=plan_info(plan);metadata[key]=(pi.route,pi.legacy_instance)
                            bench=lib.cached_i8_bench_create(plan);assert bench,lib.cached_i8_bench_error().decode();benches[key]=bench;timings[key]=[]
                    torch.cuda.synchronize()
                    keys=list(benches)
                    for sample in range(args.samples):
                        order=keys[(sample+case_index)%len(keys):]+keys[:(sample+case_index)%len(keys)]
                        if sample%2:order=order[::-1]
                        for key in order:
                            atomic(out/'active.json',dict(**case,stage='timing',candidate=key[0],output=key[1],sample=sample))
                            ms=float(lib.cached_i8_bench_run(benches[key],args.warmup,args.repeat,None));assert ms>0,lib.cached_i8_bench_error().decode();timings[key].append(ms)
                    case_records=[]
                    for key,plan in plans.items():
                        label,kind=key;atomic(out/'active.json',dict(**case,stage='verify',candidate=label,output=kind))
                        ok(lib.ck_tile_int8_nt_cached_run_v1(plan,None));torch.cuda.synchronize()
                        bad=sum(int((output[kind][i,indices].view(torch.int16)!=refs[kind][i].view(torch.int16)).sum()) for i in range(g))
                        assert bool((raw[kind][:,:16]==117).all()&(raw[kind][:,16+span:]==117).all()),('GUARD_FAIL',case,key)
                        ms=statistics.median(timings[key]);tops=2*g*m*n*k/ms/1e9
                        rec=dict(**case,candidate=label,output=kind,route=metadata[key][0],legacy_instance=metadata[key][1],ms=ms,TOPS=tops,R=tops/case['target_TOPS'] if case['target_TOPS'] else None,samples_ms=timings[key],bad_elements=bad,validation=args.validation,status='PASS' if bad==0 else 'NUMERIC_FAIL',packed_weight_bytes=wi.packed_bytes)
                        records.append(rec);case_records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
                    assert all(torch.equal(x.reshape(-1)[ix],old) for x,(ix,old) in zip([a,b,sa,sb],sampled)),('INPUT_SCALE_CHANGED',case)
                    for kind in ['BF16','FP16']:
                        winner=min((r for r in case_records if r['output']==kind and r['status']=='PASS'),key=lambda r:r['ms'])
                        default=next(r for r in case_records if r['output']==kind and r['candidate']=='default')
                        win=dict(**winner,speedup_vs_default=default['ms']/winner['ms'],default_TOPS=default['TOPS']);best.append(win);print('BEST',json.dumps(win),flush=True)
                    atomic(out/'best.json',best);atomic(out/'progress.json',dict(completed_cases=case_index+1,total_cases=len(cases),last_ID=ID,records=len(records)))
                finally:
                    for bench in benches.values():lib.cached_i8_bench_destroy(bench)
                    for plan in plans.values():ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                    ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
                del a,b,sa,sb,raw,output,refs,sampled,wd,gd,indices;torch.cuda.empty_cache()
        fields=['ID','phase','M','N','K','G','output','candidate','route','legacy_instance','ms','TOPS','target_TOPS','R','status','validation']
        with (out/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
        atomic(out/'done.json',dict(status='COMPLETE',cases=len(cases),records=len(records),numerical_failures=sum(r['status']!='PASS' for r in records)))
    except BaseException:
        atomic(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
