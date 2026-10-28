#!/usr/bin/env python3

"""Public-ABI INT8 NT verification and one-round CK/hipBLASLt comparison.



Default: representative pilot; --all explicitly selects all 1178 cases.

No input scaling, no dtype conversion, no group/shape substitution.

"""

from pathlib import Path

import argparse,csv,ctypes as C,gc,hashlib,json,os,statistics,subprocess,sys,time



def main():

    ap=argparse.ArgumentParser(description=__doc__)

    ap.add_argument('--resume',action='store_true',help='Resume only with identical binary and timing manifest')
    ap.add_argument('--all',action='store_true');ap.add_argument('--case-ids',default='1,5,6,9,35,891,1158')

    ap.add_argument('--out',required=True);ap.add_argument('--screen',action='store_true',help='CK screen vs frozen historical BLAS; not final same-round acceptance')

    ap.add_argument('--warmup',type=int,default=10);ap.add_argument('--repeat',type=int,default=50);ap.add_argument('--samples',type=int,default=3)

    ap.add_argument('--gpu',type=int,default=7);ap.add_argument('--blas-root',default='/workspace/GroupGemm/baseline/hipblaslt-install')

    ap.add_argument('--ck-lib',help='Exact shared library; defaults to the standalone CMake build')

    a=ap.parse_args();assert min(a.repeat,a.samples)>0 and a.warmup>=0

    assert os.environ.get('HIP_VISIBLE_DEVICES')==str(a.gpu) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu)

    import torch

    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False

    root=Path(__file__).resolve().parent;out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)

    assert a.resume or not (out/'records.jsonl').exists(),'Use a fresh output directory or --resume'

    base=Path('/workspace/GroupGemm/baseline/int8_nt');sys.path.insert(0,str(base))

    from bench_int8_nt import parse_shapes

    from worker import Desc

    cases=parse_shapes(base.parent/'moe_grouped_gemm_tuning_shape_final.md')

    selected=cases if a.all else [cases[int(i)-1] for i in a.case_ids.split(',')]

    historical_csv=base/'results/full_01_completed/results.csv'

    hist={int(x['ID']):x for x in csv.DictReader(historical_csv.open(encoding='utf-8-sig'))}

    configs=json.loads((base/'results/full_01/run_config.json').read_text())

    screen={x['id']:x for x in map(json.loads,(base/'results/full_01/screen.jsonl').read_text().splitlines())}

    supplement=base/'results/full_01_supplement_01/screen.jsonl'

    for line in supplement.read_text().splitlines():

        choice=json.loads(line)

        if choice.get('blas_index') is not None:screen[choice['id']]=choice

    ckpath=Path(a.ck_lib).resolve() if a.ck_lib else root/'build_cmake/libint8_nt_public.so'

    ck=C.CDLL(str(ckpath))

    ck.int8_create.argtypes=[C.POINTER(Desc),C.c_int];ck.int8_create.restype=C.c_void_p

    ck.int8_run.argtypes=[C.c_void_p,C.c_int,C.c_int];ck.int8_run.restype=C.c_float

    ck.int8_destroy.argtypes=[C.c_void_p];ck.int8_error.restype=C.c_char_p

    ck.int8_name.argtypes=[C.c_void_p];ck.int8_name.restype=C.c_char_p

    ck.int8_instance.argtypes=[C.c_void_p];ck.int8_instance.restype=C.c_uint

    ck.int8_mismatch_check.argtypes=[C.c_void_p];ck.int8_mismatch_check.restype=C.c_int

    lt=None

    if not a.screen:

        lt=C.CDLL(str(base/'build/libblas_int8.so'))

        lt.blas_create.argtypes=[C.c_int,C.POINTER(Desc)];lt.blas_create.restype=C.c_void_p

        lt.blas_count.argtypes=[C.c_void_p];lt.blas_count.restype=C.c_int

        lt.blas_select.argtypes=[C.c_void_p,C.c_int];lt.blas_select.restype=C.c_int

        lt.blas_run.argtypes=[C.c_void_p,C.c_int,C.c_int];lt.blas_run.restype=C.c_float

        lt.blas_info.argtypes=[C.c_void_p];lt.blas_info.restype=C.c_char_p

        lt.blas_destroy.argtypes=[C.c_void_p];lt.blas_error.restype=C.c_char_p;lt.blas_loaded_path.restype=C.c_char_p

        expected=(Path(a.blas_root)/'lib/libhipblaslt.so.0.10').resolve()

        assert Path(lt.blas_loaded_path().decode()).resolve()==expected

        assert hashlib.sha256(expected.read_bytes()).hexdigest()==configs['build']['fingerprint']['blas_sha256']

    binaries=[ckpath]

    if ckpath==root/'build/libint8_nt_public.so':binaries.append(root/'build/libint8_nt_probe.so')

    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

    identity={str(p):sha(p) for p in binaries}

    meta=dict(total=len(selected),gpu=a.gpu,device=torch.cuda.get_device_name(),hip=torch.version.hip,torch=torch.__version__,semantics='int8 x int8 -> int32; no scale; NT',mode='screen_vs_historical_blas' if a.screen else 'one_formal_round',warmup=a.warmup,repeat=a.repeat,samples=a.samples,binaries=identity,historical_csv=str(historical_csv),baseline_tuning_sources=[str(base/'results/full_01/screen.jsonl'),str(supplement)],shape_sha256=sha(base.parent/'moe_grouped_gemm_tuning_shape_final.md'))

    repo=root.parents[2]
    meta['source_revision']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    sources=[p for p in root.rglob('*') if p.is_file() and not any(v.startswith('build') or v in ('results','__pycache__') for v in p.relative_to(root).parts) and (p.suffix in ('.cpp','.hpp','.py','.md') or p.name=='CMakeLists.txt')]
    sources += [repo/'include/ck_tile/ops/gemm/grouped_gemm_device_args.h',repo/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp',root.parent/'19_grouped_gemm/grouped_gemm_device_args.cpp']
    meta['source_sha256']={str(p.relative_to(repo)):sha(p) for p in sources}
    if a.resume:
        previous=json.loads((out/'manifest.json').read_text())
        for key in ('binaries','semantics','mode','warmup','repeat','samples','shape_sha256','total'):
            assert previous[key]==meta[key],('Resume identity mismatch',key)
    else:(out/'manifest.json').write_text(json.dumps(meta,indent=2))

    fields=['ID','phase','dtype','layout','output dtype','scale','M','N','K','E','instance_id','instance_name','CK ms','CK TOPS','BLAS ms','BLAS TOPS','R','CK status','BLAS status','status']

    records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()] if a.resume and (out/'records.jsonl').exists() else []
    done={r['ID'] for r in records};assert len(done)==len(records)
    assert done.issubset({c['id'] for c in selected})

    def update_csv():

        with (out/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:

            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)

    for case in selected:

        assert all(sha(Path(p))==v for p,v in identity.items()),'Binary changed during run'

        if case['id'] in done:continue
        idx=case['id'];g,m,n,k=[case[key] for key in ('G','M','N','K')]

        row=dict(ID=idx,phase=case['phase'],M=m,N=n,K=k,E=g,status='RUNNING')
        row.update({'dtype':'w8a8_int8','layout':'NT','output dtype':'int32','scale':'none'})

        (out/'active.json').write_text(json.dumps(row))

        torch.manual_seed(20261026+idx)

        x=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8)

        w=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)

        raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);y=raw[:,16:-16].view(g,m,n)

        ds=(Desc*g)(*[Desc(x[e].data_ptr(),w[e].data_ptr(),y[e].data_ptr(),1,m,n,k,k,k,n,0,None,None) for e in range(g)])

        samples_in=[]

        for v in (x,w):

            ids=torch.arange(256,device='cuda',dtype=torch.int64)*(v.numel()-1)//255

            samples_in.append((ids,v.reshape(-1)[ids].clone()))

        def check():

            bad=0

            for e in range(g):

                # k*64 < 2^24 for all 1178 rows: this FP32 reference is exact.

                reference=(x[e].float()@w[e].float().T).to(torch.int32)

                bad+=int((y[e]!=reference).sum());del reference

            guards=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())

            unchanged=all(torch.equal(v.reshape(-1)[ids],old) for v,(ids,old) in zip((x,w),samples_in))

            assert bad==0 and guards and unchanged,(idx,bad,guards,unchanged)

            return dict(bad_elements=bad,output_guards=guards,input_samples_unchanged=unchanged)

        ctx=ck.int8_create(ds,g);assert ctx,ck.int8_error().decode()

        bctx=None

        try:

            assert ck.int8_mismatch_check(ctx)==-6,'Expected-instance guard failed'

            row['instance_id']=int(ck.int8_instance(ctx));row['instance_name']=ck.int8_name(ctx).decode()

            y.fill_(123456789);torch.cuda.synchronize()

            ts=[float(ck.int8_run(ctx,a.warmup,a.repeat)) for _ in range(a.samples)]

            assert min(ts)>0,(idx,ts,ck.int8_error().decode())

            row['ck_check']=check();row['CK status']='PASS';row['CK ms']=statistics.median(ts);row['ck_samples_ms']=ts

            row['CK TOPS']=case['ops']/(row['CK ms']*1e9)

            if a.screen:

                row['BLAS ms']=float(hist[idx]['BLAS ms']);row['BLAS status']='HISTORICAL_PASS'

            else:

                bctx=lt.blas_create(g,ds);assert bctx,lt.blas_error().decode()

                choice=screen[idx];index=choice.get('blas_index')

                assert index is not None,('No reusable tuned BLAS choice',idx)

                assert lt.blas_select(bctx,index)==0,lt.blas_error().decode()

                row['blas_info']=lt.blas_info(bctx).decode();assert row['blas_info']==choice['blas_info'],'BLAS tuning identity changed'

                y.fill_(123456789);torch.cuda.synchronize()

                ts=[float(lt.blas_run(bctx,a.warmup,a.repeat)) for _ in range(a.samples)]

                assert min(ts)>0,lt.blas_error().decode()

                row['blas_check']=check();row['BLAS status']='PASS';row['BLAS ms']=statistics.median(ts);row['blas_samples_ms']=ts

            row['BLAS TOPS']=case['ops']/(row['BLAS ms']*1e9);row['R']=row['BLAS ms']/row['CK ms']

            row['status']=('SCREEN_GE50' if a.screen else 'PASS') if row['R']>=.5 else 'BELOW_TARGET'

        finally:

            if bctx:lt.blas_destroy(bctx)

            ck.int8_destroy(ctx)

        records.append(row)

        with (out/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

        update_csv()

        state=dict(total=len(selected),completed=len(records),ge50=sum(r['R']>=.5 for r in records),finished=len(records)==len(selected),current_id=idx)

        (out/'progress.json').write_text(json.dumps(state,indent=2))

        print(json.dumps({k:v for k,v in row.items() if k not in ('ck_check','blas_check','blas_info')}),flush=True)

        del x,w,y,raw,ds,samples_in;gc.collect();torch.cuda.empty_cache()

    state=dict(total=len(selected),completed=len(records),ge50=sum(r['R']>=.5 for r in records),finished=len(records)==len(selected))
    (out/'progress.json').write_text(json.dumps(state,indent=2))
    print('COMPLETE',json.dumps(state),flush=True)



if __name__=='__main__':main()
