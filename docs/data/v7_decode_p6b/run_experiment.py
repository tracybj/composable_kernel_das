#!/usr/bin/env python3
"""Serial card-4 correctness and interleaved prototype measurements, resumable."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
BENCH=REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
BASE=ROOT/'bench_base'
BLAS='/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so'
ENV=dict(os.environ,HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',
    CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1',
    CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M='0')


def append(path,value):
    with path.open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def gate(key):
    for attempt in range(1,4):
        r=subprocess.run(['hy-smi'],env=ENV,text=True,stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,timeout=45)
        out=re.sub(r'\x1b\[[0-9;]*m','',r.stdout)
        rows=[line.split() for line in out.splitlines() if re.match(r'^\s*4\s',line)]
        pct=rows[0][6] if len(rows)==1 and len(rows[0])>6 else None
        append(ROOT/'gate.jsonl',dict(key=key,attempt=attempt,hcu4=pct,rc=r.returncode,output=out))
        if r.returncode==0 and 'MKFD' not in out and pct is not None:
            if pct!='0.0%':raise RuntimeError('Card 4 is busy: '+pct)
            return
        print('GATE_RETRY',key,attempt,flush=True)
        if attempt<3:time.sleep(60)
    raise RuntimeError('Gate failed three times; stop without switching GPUs')


def shapes():
    with open('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv',encoding='utf-8-sig') as f:
        return {int(r['ID']):{k:int(r[k]) for k in ['M','N','K','G']} for r in csv.DictReader(f)}


def main():
    global ROOT
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','accuracy','perf','guard','regression','profile'])
    ap.add_argument('--campaign',default='stage_wait')
    args=ap.parse_args();ROOT=ROOT/args.campaign;ROOT.mkdir(parents=True,exist_ok=True);cases=shapes();jobs=[]
    def bench(key,shape,tile=0,binary=BENCH,timing=False,extra=None,expected=None,profile=False):
        m,n,k,g=shape
        cmd=[str(binary),'--dtype=bf16','--layout=NN',f'--m={m}',f'--n={n}',f'--k={k}',f'--groups={g}',
             '--include-args-build=0',f'--correctness-repeats={3 if args.mode in ["smoke","accuracy"] else 1}',f'--backend={"ck" if timing or profile else "both"}',
             '--blas-lib='+BLAS,f'--time={int(timing or profile)}',f'--validate={int(not profile)}',
             f'--warmup={1 if profile else 10}',f'--samples={1 if profile else 30}']
        if extra:cmd+=extra
        if profile:
            cmd=['/opt/rocm-6.3.3/bin/xprof','--kernels','V7SmallM' if tile else 'V7Uniform',
                 '--sections','speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state',
                 '--output-dir','@PROFILE@']+cmd
        jobs.append(dict(key=key,command=cmd,tile=tile,expected=expected,timing=timing,profile=profile))
    if args.mode=='smoke':
        for tile in [128,64]:
            for tag,shape in [('tiny',(1,128,64,1)),('id76',(1,2048,768,4)),('tail',(65,256,80,4))]:
                bench(f'smoke_n{tile}_{tag}',shape,tile,expected=5003 if tile==128 else 5004)
    if args.mode=='accuracy':
        for tile in [128,64]:
            for m in [1,7,32,63,64,65,127,128,129,255,256]:
                bench(f'acc_n{tile}_m{m}',(m,1536,2048,4),tile,expected=5003 if tile==128 else 5004)
            for tag,shape in [('k64',(1,128,64,4)),('k80',(7,256,80,4)),('k768',(4,2048,768,4)),
                              ('k4096',(1,2048,4096,16)),('g32',(8,6144,2048,32)),('g1024',(1,128,64,1024))]:
                bench(f'acc_n{tile}_{tag}',shape,tile,expected=5003 if tile==128 else 5004)
            bench(f'acc_n{tile}_overcover',(1,256,64,4),tile,extra=['--num-cu=80'],expected=5003 if tile==128 else 5004)
            bench(f'acc_n{tile}_n1232',(1,1232,1536,4),tile,expected=4003)
            bench(f'acc_n{tile}_prefill',(512,2048,512,4),tile,expected=5000)
            bench(f'acc_n{tile}_largek',(1,4096,4112,8),tile,expected=5002)
            bench(f'acc_n{tile}_ragged',(1,2048,512,4),tile,extra=['--group-lengths=1,65,7,129'],expected=5000)
            bench(f'acc_n{tile}_empty',(1,2048,512,4),tile,extra=['--group-lengths=1,65,0,7','--allow-empty-groups=1'])
        bench('acc_n64_n64',(1,64,64,1),64,expected=5004)
        bench('acc_disable',(1,2048,768,4),128,extra=[],expected=5000)
        jobs[-1]['disable']=True
    if args.mode=='perf':
        ids=[76,681,633,1,358,942,556]
        for n,k,g,ms in [(2048,768,4,[32,128,256]),(1536,2048,4,[32,128,256]),
                         (1024,2048,8,[32,128,256]),(2048,512,8,[1,128])]:
            ids += [i for i,c in cases.items() if (c['N'],c['K'],c['G'])==(n,k,g) and c['M'] in ms]
        ids=list(dict.fromkeys(ids))
        (ROOT/'perf_ids.json').write_text(json.dumps(ids))
        for rnd in [1,2,3]:
            for i in ids:
                c=cases[i];shape=tuple(c[k] for k in ['M','N','K','G'])
                for tile in [128,64]:
                    bench(f'perf_r{rnd}_id{i}_n{tile}_before',shape,0,BASE,True,expected=5000)
                    bench(f'perf_r{rnd}_id{i}_n{tile}_after',shape,tile,BENCH,True,expected=5003 if tile==128 else 5004)
    if args.mode=='guard':
        for rnd in [1,2,3]:
            for i in [76,358,962,1118,1095,139]:
                c=cases[i];shape=tuple(c[k] for k in ['M','N','K','G'])
                for tag,binary in [('before',BASE),('after',BENCH)]:
                    bench(f'guard_r{rnd}_id{i}_{tag}',shape,0,binary,True)
    if args.mode=='profile':
        for i in [76,681,633,358]:
            c=cases[i];shape=tuple(c[k] for k in ['M','N','K','G'])
            for tile in [0,128,64]:bench(f'prof_id{i}_n{tile}',shape,tile,profile=True)
    if args.mode=='regression':
        for suffix in ['selector_test','device_args_query_test','device_args_smoke']:
            jobs.append(dict(key='reg_'+suffix,command=[str(REPO/'build/bin'/('tile_example_grouped_gemm_'+suffix))],tile=0))
        for layout in ['NN','TN']:
            jobs.append(dict(key='reg_fused_'+layout,command=[str(REPO/'build/bin/tile_example_grouped_gemm_fused_test'),
                '--layout='+layout,'--m=256','--n=2048','--k=512','--groups=16','--check=1','--cpu-ref=1','--iters=1','--warmup=1'],tile=0))
    with (BASE.parent/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=ROOT/'jobs.jsonl'
        done={r['key'] for s in journal.read_text().splitlines() if (r:=json.loads(s)).get('ok')} if journal.exists() else set()
        for spec in jobs:
            if spec['key'] in done:continue
            gate(spec['key'])
            parent=ROOT/'runs'/spec['key'];parent.mkdir(parents=True,exist_ok=True)
            folder=parent/str(len(list(parent.iterdir()))+1);folder.mkdir()
            cmd=[s.replace('@PROFILE@',str(folder/'profile')) for s in spec['command']]
            env=dict(ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M=str(spec['tile']),
                     LD_LIBRARY_PATH=str(Path(BLAS).parent)+':'+ENV.get('LD_LIBRARY_PATH',''))
            if spec.get('disable'):env['CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M']='1'
            if args.mode=='regression':
                env['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN']='1' if 'fused' in spec['key'] else '0'
                env['CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN']='1' if spec['key']=='reg_fused_TN' else '0'
            record=dict(spec,path=str(folder),command=cmd,at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
            print('START',spec['key'],flush=True);t=time.monotonic()
            with (folder/'run.log').open('w') as log:
                proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=900)
            out=(folder/'run.log').read_text(errors='replace');record.update(rc=proc.returncode,seconds=time.monotonic()-t,ok=proc.returncode==0)
            selected=re.search(r'SELECT instance_id=(\d+)',out)
            if selected:record['selected']=int(selected[1])
            if spec.get('expected') is not None:record['ok'] &= record.get('selected')==spec['expected']
            if spec.get('timing'):
                perf=re.search(r'PERF backend=ck_device_args (.*)',out)
                record['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',perf[1])} if perf else None
                record['ok'] &= bool(perf) and 'SUMMARY result=PASS' in out
            if spec.get('profile'):
                record['perf_files']=[str(p) for p in (folder/'profile').glob('*.perf')]
                record['ok'] &= len(record['perf_files'])==1
            elif args.mode in ['smoke','accuracy']:
                record['ok'] &= bool(re.search(r'ACCURACY backend=ck_vs_blas .*?failed=0 max_abs=0(?:\s|$)',out))
            append(journal,record)
            print('DONE',spec['key'],'ok='+str(record['ok']),'sec='+str(round(record['seconds'],1)),flush=True)
            if not record['ok']:raise RuntimeError('Stopped; inspect '+str(folder/'run.log'))
    print('ALL_DONE',args.mode,flush=True)


if __name__=='__main__':main()
