"""Audit this authorized continuation from raw evidence; no GPU or code mutation."""
from pathlib import Path
import collections,datetime,hashlib,json,subprocess,tarfile,time,sys,difflib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import runtime as r,plan
REPO=Path('/workspace/composable_kernel')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def journal(name):
    p=ROOT/name
    if not p.exists():return []
    return [json.loads(x) for x in p.read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]

def main():
    session=r.read(ROOT/'session.json')
    jobs=journal('jobs.jsonl');started=journal('started.jsonl')
    gates={x['key']:x for x in journal('gates.jsonl')}
    kernels={x['key']:x for x in journal('kernel_checks.jsonl')}
    keys={x['key'] for x in jobs};assert len(keys)==len(jobs)
    startkeys={x['record']['key'] for x in started};assert len(startkeys)==len(started)
    assert keys<=startkeys
    issues=[];retained_failed_profiles=[];comparisons=[];groups={};verified_versions=set();manifest_cache={}
    for rec in jobs:
        try:
            assert rec['physical_gpu']==4
            assert rec['environment']['HIP_VISIBLE_DEVICES']==rec['environment']['CUDA_VISIBLE_DEVICES']=='4'
            at=datetime.datetime.fromisoformat(rec['at']).timestamp()
            assert session['t0_epoch']<=at<session['gpu_deadline_epoch']
            gate=gates[rec['key']]
            assert gate['rc']==0 and gate['hcu']=='0.0%' and 'MKFD' not in gate['output'].upper()
            assert gate['mem_available_kib']>=64*1024*1024
            assert all(kernels[rec['key']+s]['rc']==0 and not kernels[rec['key']+s]['events'] for s in ['_pre','_post'])
            logpath=Path(rec['path'])/'run.log'
            if not rec['ok']:
                assert rec['key']=='e0_h3wave4_id686_base_cu15_bd384a9a_5095deded4'
                assert rec['kind']=='profile' and rec['rc']==134 and rec['parse_error']=='Profiled app failed'
                assert 'Aborted' in logpath.read_text()
                assert (ROOT/'wave4_profile_abort_classification.json').exists()
                retained_failed_profiles.append(dict(key=rec['key'],rc=rec['rc'],kind=rec['kind'],raw_log=str(logpath),raw_log_sha256=sha(logpath),excluded_from_valid_measurements=True,cause='Profiler-associated application abort; cause not established as MKFD'))
                parsed={}
            else:
                assert rec['rc']==0
                parsed=r.parse_output(logpath.read_text(),rec['kind'],rec.get('expected'),rec.get('repeats',1),rec.get('regression_kind'))
            if rec['kind']=='accuracy':
                assert [list(x) for x in parsed['comparisons']]==rec['comparisons']
                assert all(int(x[1])==rec['shape']['M']*rec['shape']['N']*rec['shape']['G'] and int(x[2])==0 for x in parsed['comparisons'])
                comparisons+=parsed['comparisons']
            if rec['kind']=='perf':assert parsed['result']==rec['result']
            if rec['plan_sha256'] not in manifest_cache:
                manifest_cache[rec['plan_sha256']]=r.read(ROOT/'manifests'/(rec['plan_sha256']+'.json'))
            manifest=manifest_cache[rec['plan_sha256']]
            assert manifest['benchmark_sha256']==rec['benchmark_sha256']
            canonical=dict(manifest);canonical.pop('plan_sha256')
            assert plan.digest(canonical)==rec['plan_sha256']
            for name,digest in rec['script_sha256'].items():
                if (name,digest) not in verified_versions:
                    assert sha(ROOT/'source_versions'/(digest+'_'+name))==digest
                    verified_versions.add((name,digest))
            group=groups.setdefault(rec['benchmark_sha256'],dict(jobs=0,kinds=collections.Counter(),epochs=set(),manifest_shas=set()))
            group['jobs']+=1;group.setdefault('failed_profile_jobs',0);group['failed_profile_jobs']+=int(not rec['ok']);group['kinds'][rec['kind']]+=1;group['epochs'].add(rec['epoch']);group['manifest_shas'].add(rec['plan_sha256'])
        except Exception as exc:
            issues.append(dict(key=rec['key'],kind=rec['kind'],error=type(exc).__name__+': '+str(exc),raw_path=rec['path']))
    for group in groups.values():
        group['kinds']=dict(group['kinds']);group['epochs']=sorted(group['epochs']);group['manifest_shas']=sorted(group['manifest_shas'])
    deliveries=[]
    for receipt in sorted(ROOT.glob('profile_delivery_batch*.json')):
        entry=r.read(receipt);package=Path(entry['path']);assert package.stat().st_size==entry['bytes'] and sha(package)==entry['sha256']
        count=0;captured=0
        with tarfile.open(package) as archive:
            mf=next(x for x in archive.getmembers() if x.name.endswith('/profile_manifest.json'))
            prefix=mf.name.rsplit('/',1)[0];manifest=json.loads(archive.extractfile(mf).read());captured=len(manifest)
            for cap in manifest:
                assert cap['physical_gpu']==4 and cap['sqtt'] and cap['raw_token_bytes']>=4096
                assert cap['selected_dispatch']==18
                for item in cap['files']:
                    data=archive.extractfile(prefix+'/'+item['path']).read()
                    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'];count+=1
        deliveries.append(dict(receipt=str(receipt),receipt_sha256=sha(receipt),package=str(package),package_sha256=entry['sha256'],captures=captured,compressed_capture_members_SHA_verified=count,GUI_verified=False))
    initial=r.read(ROOT/'initial_files_sha256.json')
    untracked=[x[3:] for x in (ROOT/'initial_status.txt').read_text().splitlines() if x.startswith('?? ')]
    assert len(untracked)==10 and all(sha(REPO/n)==initial[n] for n in untracked)
    rf='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'
    diff=subprocess.check_output(['git','diff','--',rf],cwd=REPO)
    assert hashlib.md5(diff).hexdigest()==session['protected_diff_md5']
    assert sha(session['previous_session'])==session['previous_session_sha256']
    assert sha(REPO/'docs/data/decode_depth8_replay_p6k/results.json')=='00c80c665735d59165947e5c9458e4e3cb0327df1a47dee2926fe725af4e9d3d'
    for p in ['/workspace/shapes_1178.csv','/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv']:
        assert sha(p)=='5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
    phases=[]
    for name in ['phase','k64','n256','b_reuse','low_lds','wave4','noslc','guard','accum']:
        path=ROOT/(name+'_discovery_analysis.json')
        if path.exists():
            analysis=r.read(path);phases.append(dict(name=name,path=str(path),sha256=sha(path),completed_pairs=analysis['completed_pairs'],groups=analysis['groups'],scope='Each phase separately same-binary; three-round discovery never merged into a per-shape best-of-build table'))
    controllers={}
    for name in ['noslc_controller_status.json','guard_controller_status.json','accum_controller_status.json','compile/build17/controller.json']:
        p=ROOT/name
        if p.exists():controllers[name]=r.read(p)
    final=None
    if (ROOT/'final_active.json').exists():
        folder=Path(r.read(ROOT/'final_active.json')['path']);p=folder/'report.json'
        if p.exists():
            report=r.read(p);final=dict(path=str(p),sha256=sha(p),goal_performance_pass=report['goal_performance_pass'],stable_weak_originals=report['stable_weak_originals'],attributable_crossings=report['attributable_crossings'])
    report=dict(at=r.stamp(),CPU_only=True,GPU=False,session=str(ROOT/'session.json'),session_sha256=sha(ROOT/'session.json'),
        audit_scope='Completed physical records only; pending starts remain pending. This script never chooses candidates or changes acceptance.',
        started=len(started),completed=len(jobs),retained_failed_profiles=retained_failed_profiles,all_submissions_succeeded=not retained_failed_profiles,valid_accuracy_and_perf_records_verified=not issues,pending_job_keys=sorted(startkeys-keys),issues=issues,completed_records_raw_audit_ok=not issues,
        physical_gpu=4,by_binary=groups,CK_full_comparisons=len(comparisons),CK_full_elements=sum(int(x[1]) for x in comparisons),
        CK_failed_elements=sum(int(x[2]) for x in comparisons),profile_deliveries=deliveries,phases=phases,controllers=controllers,
        initial_untracked10_SHA_verified=True,readfirstlane_preserved=True,old_session_unchanged=True,frozen_P6k_and_1193_shapes_unchanged=True,
        recovery=r.read(ROOT/'recovery_total.json'),environment_epoch=r.read(ROOT/'environment_epoch.json'),
        final_confirmation=final,Goal_complete=False,goal_completion_requires_all_task_deliverables_and_actual_acceptance=True,
        new_experiment_cutoff_UTC=datetime.datetime.fromtimestamp(session['gpu_deadline_epoch'],datetime.timezone.utc).isoformat(),
        hard_end_UTC=datetime.datetime.fromtimestamp(session['deadline_epoch'],datetime.timezone.utc).isoformat(),
        performance_definition='P6k frozen GPU0 Native BLAS remains unchanged; all continuation CK paired measurements are GPU4. Fresh domain, synthetic cases and mixed historical1193 projection must remain distinct.')
    r.write(ROOT/'readonly_reports/final_continuation_raw_audit.json',report)
    print('CONTINUATION_READONLY_RAW_AUDIT',len(jobs),'pending',len(startkeys-keys),'issues',len(issues),'full',len(comparisons),'perfpackages',len(deliveries),'retained_failed_profiles',len(retained_failed_profiles),flush=True)
    if issues:raise SystemExit(1)

if __name__=='__main__':main()
