#!/usr/bin/env python3
"""Counter-only follow-up for newly observed square128 regressions."""
import argparse,datetime,fcntl,importlib.util,json,re,resource,subprocess,sys,time
from pathlib import Path
import health,journalview
import run as r

p=r.ROOT

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser();parser.add_argument('--repair-epoch',default='rec3');args=parser.parse_args()
    plan_path=p/'square_diagnostic_plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(dict(ids=[934,714],variants=['base','square'],planned_at=r.stamp(),benchmark_sha256=r.sha(r.BENCH),purpose='Diagnose representative M128/M256 square regressions found in predeclared100-case scope; counters only, no new performance claim'),indent=2))
    plan=json.loads(plan_path.read_text());assert plan['benchmark_sha256']==r.sha(r.BENCH)
    deadline=datetime.datetime.fromisoformat(json.loads((p/'session.json').read_text())['deadline_utc']).timestamp()
    with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,good=journalview.read()
        for i in plan['ids']:
            for tag,variant in [('before','base'),('after','square')]:
                logical=f'squarediag_id{i}_{tag}';key=args.repair_epoch+'_'+logical
                if logical in good:continue
                assert r.sha(r.BENCH)==plan['benchmark_sha256']
                if time.time()>deadline-900:raise RuntimeError('Deadline reserve')
                health.gate(key)
                c=r.cases()[i];folder=p/'runs'/key/'1';folder.mkdir(parents=True,exist_ok=False)
                env=dict(r.ENV,CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='0',CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='3' if variant=='square' else '0',CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='0')
                cmd=['/opt/rocm-6.3.3/bin/xprof','--kernels','V7Uniform','--sections',r.GC,'--output-dir',str(folder/'profile'),str(r.BENCH),'--dtype=bf16','--layout=NN',*[f'--{x}={c[y]}' for x,y in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--backend=ck','--time=1','--validate=0','--include-args-build=0','--warmup=1','--samples=1']
                rec=dict(key=key,logical_key=logical,id=i,shape=c,variant=variant,kind='profile',command=cmd,path=str(folder),benchmark_sha256=r.sha(r.BENCH),script_sha256=r.sha(__file__),environment_epoch=r.current_epoch(),at=r.stamp(),environment={k:v for k,v in env.items() if k.startswith(('CK_TILE_GROUPED_GEMM_','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))})
                (folder/'command.json').write_text(json.dumps(rec,indent=2));start=time.monotonic()
                try:
                    with (folder/'run.log').open('w') as output:
                        result=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=300)
                    text=(folder/'run.log').read_text();selected=re.search(r'SELECT instance_id=(\d+)',text)
                    files=[str(path) for path in (folder/'profile').glob('*.perf')]
                    rec.update(rc=result.returncode,seconds=time.monotonic()-start,selected=int(selected[1]) if selected else None,perf_files=files)
                    rec['ok']=result.returncode==0 and rec['selected']==(5009 if variant=='square' else 5000) and len(files)==1 and 'SUMMARY result=PASS' in text
                except Exception as error:
                    rec.update(ok=False,error=repr(error),seconds=time.monotonic()-start)
                r.append(p/'jobs.jsonl',rec);print(key,rec['ok'],flush=True)
                if not rec['ok']:raise RuntimeError('STOP: '+str(folder/'run.log'))
        _,good=journalview.read()
        sys.path.insert(0,'/workspace/v7_coverage/p6d')
        spec=importlib.util.spec_from_file_location('p6d_analysis','/workspace/v7_coverage/p6d/analyze.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
        rows=[]
        for i in plan['ids']:
            pair=[good[f'squarediag_id{i}_{tag}'] for tag in ['before','after']]
            assert pair[0]['environment_epoch']==pair[1]['environment_epoch']
            for record in pair:
                square=record['variant']=='square';tile=(128,128,16) if square else (256,256,16)
                dispatches=d.dispatches(record['perf_files'][0]);geometry=d.dispatch_info(record['perf_files'][0]);shape=tuple(record['shape'][k] for k in ['M','N','K','G']);metrics=d.p6.metrics(dispatches[-1],shape,tile)
                assert len(dispatches)==len(geometry)==2
                for dispatch,geo in zip(dispatches,geometry):
                    assert dispatch['SQ_INSTS_MMOP']==metrics['mmop_insts_expected_padded']
                    assert dispatch['SQ_WAVES']==metrics['ctas']*(4 if square else 8) and geo['ctas']==metrics['ctas']
                    assert geo['lds']==(32768 if square else 65536) and geo['scratch']==0
                rows.append(dict(id=i,variant=record['variant'],command_key=record['key'],source=record['perf_files'][0],environment_epoch=record['environment_epoch'],geometry=geometry,raw_dispatches=dispatches,**metrics))
        (p/'square_diagnostic_profiles.json').write_text(json.dumps(rows,indent=2))
        print('SQUARE_DIAGNOSTIC_PASS',len(rows),'profiles',2*len(rows),'dispatches')

if __name__=='__main__':main()
