#!/usr/bin/env python3
"""One-command CSV replay. Run --dry-run before serial GPU execution."""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
import health
import plan
import report
import runtime as r
import qualify

PILOT = [1179,1180,1181,1182,1183,1184,146,620,688,781,1186,1193]
REGRESSIONS = ['selector_test', 'decode_depth8_selector_test', 'device_args_query_test', 'device_args_smoke']
PROTECTED = 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'


def new_session():
    now=time.time(); z=datetime.timezone(datetime.timedelta(hours=8))
    return dict(t0_epoch=now,t0_utc=datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat(),
                t0_beijing=datetime.datetime.fromtimestamp(now,z).isoformat(),t0_monotonic=time.monotonic(),
                deadline_epoch=now+43200,gpu_deadline_epoch=now+36000,
                new_features_deadline_epoch=now+36000,physical_gpu=4,
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan.REPO,text=True).strip())


def prepare(args):
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError('Output exists: use --resume; never overwrite an earlier run')
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'session.json').exists(): r.write(root/'session.json',new_session())
    session=r.read(root/'session.json')
    if session['physical_gpu']!=4 or session['deadline_epoch']-session['t0_epoch']>43200.01:
        raise ValueError('Session violates GPU4/12h authorization')
    table=plan.load_table(args.plan)
    if args.depth8:
        old_bins=report.bins(table['cases'],historical=True)
        if old_bins['prefill']['bins']!=[293,19,0,0,0,0] or old_bins['decode']['bins']!=[610,257,14,0,0,0]:
            raise ValueError('Input does not match the user\'s original policy table buckets')
        table=plan.with_depth8_overlay(table)
    check=plan.selector_check(table,root/'cpu_primary')
    audits=[dict(csv_path=table['csv_path'],csv_sha256=table['csv_sha256'],**check)]
    for source in args.check_table:
        other=plan.load_table(source)
        audits.append(dict(csv_path=other['csv_path'],csv_sha256=other['csv_sha256'],
                           **plan.selector_check(other,root/('cpu_'+other['csv_sha256'][:12]))))
    r.write(root/'selector_audit.json',dict(at=r.stamp(),cpu_only=True,checks=audits,total=sum(x['cases'] for x in audits)))
    p=root/'frozen_plan.json'
    if p.exists():
        manifest=r.read(p)
        if manifest['table']!=table: raise ValueError('CSV/recipe changed; use a new authorized run, not in-place resume')
        for path,sha in [(plan.BENCH,manifest['benchmark_sha256']),(plan.BLAS,manifest['blas_sha256']),
                         (plan.SELECTOR,manifest['selector_sha256'])]:
            if plan.sha(path)!=sha: raise ValueError('Frozen executable/library/selector changed: '+str(path))
        for path,sha in manifest.get('worktree_source_sha256',{}).items():
            if plan.sha(plan.REPO/path)!=sha:raise ValueError('Frozen build source changed: '+path)
        for path,sha in manifest.get('dependencies',{}).get('sha256',{}).items():
            if plan.sha(path)!=sha:raise ValueError('Frozen DSO changed: '+path)
    else:
        diff=subprocess.check_output(['git','diff','--',PROTECTED],cwd=plan.REPO)
        (root/'protected_user.patch').write_bytes(diff)
        manifest=dict(table=table,benchmark=str(plan.BENCH),benchmark_sha256=plan.sha(plan.BENCH),
                      blas_library=str(plan.BLAS),blas_sha256=plan.sha(plan.BLAS),
                      selector_sha256=plan.sha(plan.SELECTOR),
                      protected_diff_md5=hashlib.md5(diff).hexdigest(),probe=health.probe_evidence(),
                      pilot_ids=PILOT,physical_gpu=4,
                      protocol=dict(dtype='bf16',layout='NN',warmup=10,samples=30,launches_per_sample=1,
                                    include_args_build=0,rounds=3,correctness_repeats=1,accuracy_redzone=16,
                                    formal_timing_redzone=0,OMP_NUM_THREADS=4),
                      regression_binary_sha256={s:plan.sha(plan.REPO/'build/bin'/('tile_example_grouped_gemm_'+s)) for s in REGRESSIONS})
        manifest['qualification_plan']=qualify.prepare(table,root/'cpu_qualification') if args.depth8 else None
        manifest['runner_sha256']={p.name:plan.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        sources=['include/ck_tile/ops/gemm/grouped_gemm_selector.hpp',
                 'include/ck_tile/ops/gemm/grouped_gemm_device_args.h',
                 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_depth8.hpp',
                 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp',
                 'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp',
                 'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_device_args_provider.hpp',
                 'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_depth8_device_args.cpp',
                 'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_decode_device_args_provider_sources.txt',
                 'example_hcu/ck_tile/19_grouped_gemm/CMakeLists.txt']
        manifest['worktree_source_sha256']={name:plan.sha(plan.REPO/name) for name in sources}
        manifest['dependencies']=plan.dependencies()
        manifest['plan_sha256']=plan.digest(manifest)
        r.write(p,manifest)
        (root/'input.csv').write_bytes(Path(args.plan).read_bytes())
        subprocess.run(['ldd',str(plan.BENCH)],stdout=(root/'ldd.log').open('w'),stderr=subprocess.STDOUT,check=True,timeout=30)
    if manifest['plan_sha256']!=plan.digest({k:v for k,v in manifest.items() if k!='plan_sha256'}):
        raise ValueError('Frozen manifest digest mismatch')
    print('PLAN_VERIFIED cases1193 primary_selector1193 additional_selector'+str(sum(x['cases'] for x in audits[1:])),flush=True)
    return r.Site(root)


def run_once(site,pilot_only=False,qualification_only=False):
    with r.locked():
        if site.stop.exists(): raise r.QueueStopped('Stop marker exists; no new GPU submissions')
        if site.mark.exists(): raise r.EnvironmentFailure('Pending incident requires recovery')
        raw=report.records(site.root)
        for suffix in REGRESSIONS:
            logical='regression_'+suffix
            old=[x for x in raw if x.get('ok') and x.get('logical_key')==logical
                 and x['plan_sha256']==site.manifest['plan_sha256']]
            if old: continue
            site.execute(dict(logical_key=logical,kind='regression',regression_kind=suffix,
                              env={plan.V7:'0'},command=[str(plan.REPO/'build/bin'/('tile_example_grouped_gemm_'+suffix))],
                              expected_binary_sha256=site.manifest['regression_binary_sha256'][suffix]))
        if site.manifest.get('qualification_plan'):
            qualify.run(site)
        if qualification_only:return
        done,partial=report.completed(site.root,site.manifest)
        cases={x['ID']:x for x in site.manifest['table']['cases']}
        order=PILOT+([] if pilot_only else [i for i in sorted(cases) if i not in PILOT])
        for i in order:
            if i in done: continue
            case=cases[i];attempt=uuid.uuid4().hex[:16]
            common=dict(ID=i,shape={k:case[k] for k in ['M','N','K','G']},expected=case['expected'],
                        env=case['env'],attempt=attempt)
            site.execute(dict(common,logical_key=f'id{i}_accuracy',kind='accuracy',round=0,
                              repeats=case.get('accuracy_repeats',1),
                              command=plan.command(case,'accuracy')))
            for rd in [1,2,3]:
                site.execute(dict(common,logical_key=f'id{i}_r{rd}',kind='perf',round=rd,
                                  command=plan.command(case,'perf')))
            # Persist completion independently of generated CSV/report files.
            done[i]={'attempt':attempt}
            site.events('shape_completions.jsonl',ID=i,epoch=site.epoch(),attempt=attempt,completed=len(done))
            site.status(state='shape_complete',ID=i,completed=len(done),expected=1193,epoch=site.epoch())
            print('SHAPE_COMPLETE',i,len(done),'/1193',flush=True)
            if set(PILOT)<=set(done) and not (site.root/'pilot_complete.json').exists():
                summary=report.emit(site.root)
                r.write(site.root/'pilot_complete.json',dict(at=r.stamp(),IDs=PILOT,completed=summary['accepted_shapes']))
                print('PILOT_PASS continuing frozen1193 plan',flush=True)
            if len(done)%25==0: report.emit(site.root)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',help='Source-labelled1193 CSV; required except recovery/report-only')
    ap.add_argument('--output',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--dry-run',action='store_true',help='CPU parsing/selector only; never invokes GPU or hy-smi')
    ap.add_argument('--check-table',action='append',default=[],help='Additional CPU-only1193 table')
    ap.add_argument('--depth8',action='store_true',help='Apply the six-shape property overlay')
    ap.add_argument('--pilot-only',action='store_true',help='Qualify and measure the frozen pilot subset')
    ap.add_argument('--qualification-only',action='store_true',help='Run contract accuracy and six A/B pairs only')
    ap.add_argument('--recover',action='store_true',help=argparse.SUPPRESS)
    ap.add_argument('--report-only',action='store_true')
    args=ap.parse_args(argv);r.no_core()
    if args.recover:
        health.recover(r.Site(args.output));return 0
    if args.report_only:
        print(json.dumps(report.emit(args.output)));return 0
    if not args.plan: ap.error('--plan is required')
    site=prepare(args)
    if args.dry_run:
        site.status(state='dry_run_passed',cpu_only=True)
        print('DRY_RUN_PASS no GPU commands',flush=True);return 0
    try:
        while True:
            try:
                run_once(site,args.pilot_only,args.qualification_only);break
            except r.EnvironmentFailure as ex:
                site.status(state='environment_recovery_required',reason=str(ex))
                report.emit(site.root)
                marker=r.read(site.mark)
                name='recovery_controller_'+uuid.uuid4().hex[:12]+'.log'
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(site.root),'--recover']
                site.events('controller_events.jsonl',event='fresh_recovery_process',command=cmd,marker=marker,log=name)
                with (site.root/name).open('w') as log:
                    p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,site.gpu_remaining()-30))
                if p.returncode: raise r.QueueStopped('Recovery failed; see '+name)
        summary=report.emit(site.root)
        if args.qualification_only:
            site.status(state='qualification_complete',completed=summary['accepted_shapes'])
            print('QUALIFICATION_STAGE_PASS',flush=True);return 0
        if args.pilot_only:
            done,_=report.completed(site.root,site.manifest)
            if not set(PILOT)<=set(done):raise ValueError('Pilot incomplete')
            site.status(state='pilot_complete',completed=len(done))
            print('PILOT_STAGE_PASS',flush=True);return 0
        if not summary['complete']: raise ValueError('Controller ended without all1193 accepted shapes')
        site.status(state='GPU_complete_pending_final_audit',completed=1193)
        print('FULL_REPLAY_GPU_PASS shapes1193',flush=True);return 0
    except r.QueueStopped as ex:
        site.status(state='GPU_queue_stopped',reason=str(ex));print('GPU_STOP',str(ex),flush=True);return 2
    finally:
        report.emit(site.root)


if __name__=='__main__':
    raise SystemExit(main())
