"""Run a frozen, finite job plan serially; physical records never overwritten."""
import argparse,csv,json,sys,time
from pathlib import Path
import runtime as r
BENCH=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
def cases():
    return {int(x['ID']):dict(id=int(x['ID']),**{k:int(x[k]) for k in ['M','N','K','G']}) for x in csv.DictReader(open('/workspace/shapes_1178.csv',encoding='utf-8-sig'))}
def bench(key,c,expected,overrides=None,kind='accuracy',repeat=3):
    timing=kind in ['perf','profile'];profile=kind=='profile'
    cmd=[str(BENCH),'--dtype=bf16','--layout=NN',*[f'--{a}={c[k]}' for a,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--backend='+('ck' if timing else 'both'),'--blas-lib='+r.BLAS,'--include-args-build=0',f'--time={int(timing)}',f'--validate={int(not profile)}',f'--warmup={1 if profile else 10}',f'--samples={1 if profile else 30}',f'--correctness-repeats={1 if timing else repeat}']
    return dict(key=key,shape=c,id=c['id'],expected=expected,command=cmd,env=overrides or {},kind=kind)
def cache_plan(suite):
    cs=cases();jobs=[]
    variants=[(m,n,5000 if m==1 and n==0 else 5005 if m==1 else 5002 if m==8 and n==0 else 5006 if m==8 else 5011 if n==0 else 5012) for m in [1,5,8] for n in [0,1]]
    def ev(m,n):return dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK=str(m),CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=str(n))
    if suite=='cache_accuracy':
        for i in [781,782,783,784]:
            for m,n,num in variants:jobs.append(bench(f'acc_cache_id{i}_m{m}_n{n}',cs[i],num,ev(m,n),repeat=1))
        # M tile counts 1/2/7/8/9/17 exercise partial M01=5 strips.
        for mt in [1,2,7,8,9,17]:
            c=dict(id=-mt,M=256*(mt-1)+1,N=256*3,K=80,G=2)
            for n in [0,1]:jobs.append(bench(f'acc_cache_strip{mt}_n{n}',c,5011+n,ev(5,n)))
    elif suite=='cache_perf':
        for rnd in [1,2,3]:
            for i in [781,782,783,784]:
                for m,n,num in variants:
                    if (m,n)==(1,0):continue
                    pair=f'cache_r{rnd}_id{i}_m{m}_n{n}'
                    for arm,em,en,eid in [('control',1,0,5000),('candidate',m,n,num)]:
                        spec=bench(pair+'_'+arm,cs[i],eid,ev(em,en),'perf');spec.update(round=rnd,pair=pair,arm=arm,variant=f'm{m}_n{n}');jobs.append(spec)
    elif suite=='regression':
        for suffix in ['selector_test','device_args_query_test','device_args_smoke']:
            jobs.append(dict(key='regression_'+suffix,regression_kind=suffix,command=[str(r.REPO/'build/bin'/('tile_example_grouped_gemm_'+suffix))],kind='regression',env=dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='0')))
    elif suite=='cache_native':
        for rnd in [1,2,3]:
            for i in [781,782,783,784]:
                script=r.ROOT/'native_replay.py'
                jobs.append(dict(key=f'cache_native_r{rnd}_id{i}',id=i,round=rnd,kind='native_perf',replay_script_sha256=r.sha(script),command=[sys.executable,'-u',str(script),f'--id={i}','--warmup=10','--samples=30','--repeats=1,20','--validate=0'],env=dict(LD_PRELOAD=r.BLAS)))
    return jobs
def main():
    r.no_core();ap=argparse.ArgumentParser();ap.add_argument('suite');ap.add_argument('--repair-tag',default='');ap.add_argument('--plan');args=ap.parse_args()
    jobs=json.loads(Path(args.plan).read_text())['jobs'] if args.plan else cache_plan(args.suite);assert jobs
    plan=r.ROOT/(args.suite+'_plan.json')
    data=dict(at=r.stamp(),suite=args.suite,benchmark_sha256=r.sha(BENCH),scope='fixed representatives; diagnostic factor isolation; no production attribute domain inferred from winners',execution_policy='non-timing first, then complete each(shape,variant) three-round contrast before next shape; same jobs/protocol',jobs=jobs)
    if plan.exists():
        old=json.loads(plan.read_text());assert old['jobs']==jobs and old['benchmark_sha256']==data['benchmark_sha256'],'Plan/binary changed; use new campaign name'
    else:plan.write_text(json.dumps(data,indent=2))
    with r.locked():
        done=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()] if (r.ROOT/'jobs.jsonl').exists() else []
        valid=[x for x in done if x.get('ok') and x.get('campaign_binary_sha256')==data['benchmark_sha256']]
        current={(x.get('logical_key',x['key']),x['epoch']) for x in valid}
        # A recovery must redo every round/arm of a partially affected contrast.
        # Fully completed contrasts from an earlier healthy epoch stay valid.
        def group(spec):
            if spec.get('kind') in ['perf','native_perf']:
                return (spec.get('kind'),spec.get('id'),spec.get('variant','default'))
            return ('single',spec['key'])
        groups={}
        for spec in jobs:groups.setdefault(group(spec),set()).add(spec['key'])
        complete=set()
        for g,keys in groups.items():
            for ep in {x['epoch'] for x in valid}:
                available={x.get('logical_key',x['key']) for x in valid if x['epoch']==ep}
                if keys<=available:complete.update(keys)
        # Close each complete contrast before moving on; the original frozen
        # job set/protocol stays intact. Completed shapes survive later service
        # interruptions without repeating an entire round over all shapes.
        timing=[x for x in jobs if x.get('kind') in ['perf','native_perf']]
        other=[x for x in jobs if x.get('kind') not in ['perf','native_perf']]
        timing.sort(key=lambda x:(x['id'],x.get('variant','default'),x['round'],0 if x.get('arm','control')=='control' else 1))
        execution=other+timing
        assert len(execution)==len(jobs) and {x['key'] for x in execution}=={x['key'] for x in jobs}
        r.append(r.ROOT/'execution_schedules.jsonl',dict(at=r.stamp(),suite=args.suite,epoch=r.epoch(),benchmark_sha256=data['benchmark_sha256'],repair_tag=args.repair_tag,policy=data['execution_policy'],keys=[x['key'] for x in execution]))
        for spec0 in execution:
            spec=dict(spec0); logical=spec['key'];spec['logical_key']=logical
            spec['campaign_binary_sha256']=data['benchmark_sha256']
            if logical in complete or (logical,r.epoch()) in current:continue
            # Refuse late jobs rather than terminating an in-flight HIP process.
            if r.remaining(new=True)<spec.get('expected_max_seconds',180)+60:raise RuntimeError('NEW_EXPERIMENT_CUTOFF_RESERVE')
            assert r.sha(BENCH)==data['benchmark_sha256'],'Binary changed mid campaign'
            if args.repair_tag:spec['key']=args.repair_tag+'_'+spec['key']
            spec['key']='e'+str(r.epoch())+'_'+spec['key']
            if spec.get('kind')=='profile':
                folder=r.ROOT/'runs'/spec['key']
                spec['binary']=spec.get('binary',spec['command'][0])
                spec['command']=['/opt/rocm-6.3.3/bin/xprof','--kernels',spec.get('kernel_filter','V7Uniform'),'--sections','speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state','--output-dir',str(folder/'profile')]+spec['command']
            r.execute(spec)
    print('CAMPAIGN_DONE',args.suite,flush=True)
if __name__=='__main__':main()
