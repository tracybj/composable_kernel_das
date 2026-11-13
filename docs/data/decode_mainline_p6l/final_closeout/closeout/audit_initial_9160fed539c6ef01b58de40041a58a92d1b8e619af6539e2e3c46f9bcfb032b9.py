"""CPU-only closeout audit of existing evidence; submits no GPU or compiler work."""
from pathlib import Path
import collections,datetime,hashlib,json,math,statistics,tarfile,time
import acceptance,plan,runtime as r
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'closeout'

def journal(path):return [json.loads(s) for s in path.read_text().splitlines() if s]

def main():
    session=r.read(ROOT/'session.json');assert session['gpu_deadline_epoch']<time.time()<session['deadline_epoch']
    OUT.mkdir(exist_ok=True)
    jobs=journal(ROOT/'jobs.jsonl');bykey={x['key']:x for x in jobs};assert len(jobs)==len(bykey)==6632
    started=journal(ROOT/'started.jsonl');assert {x['record']['key'] for x in started}==set(bykey) and len(started)==len(jobs)
    gates={x['key']:x for x in journal(ROOT/'gates.jsonl')};kernels={x['key']:x for x in journal(ROOT/'kernel_checks.jsonl')}
    ck_full=ck_elements=native_full=native_elements=0;raw=[];versions={};manifests={}
    for rec in jobs:
        key=rec['key'];assert rec['ok'] and rec['rc']==0 and rec['physical_gpu']==4
        assert rec['environment']['HIP_VISIBLE_DEVICES']==rec['environment']['CUDA_VISIBLE_DEVICES']=='4'
        assert datetime.datetime.fromisoformat(rec['at']).timestamp()<session['gpu_deadline_epoch']
        assert not rec['kernel_service_events']
        for suffix in ['','_post']:
            gate=gates[key+suffix];assert gate['rc']==0 and gate['hcu']=='0.0%' and gate['mem_available_kib']>=64*1024*1024
            assert 'MKFD' not in gate['output'].upper()
        for suffix in ['_pre','_post']:
            check=kernels[key+suffix];assert check['rc']==0 and not check['events']
        manifest=manifests.setdefault(rec['plan_sha256'],r.read(ROOT/'manifests'/(rec['plan_sha256']+'.json')))
        assert manifest['benchmark_sha256']==rec['benchmark_sha256']
        assert all(rec['script_sha256'][n]==sha for n,sha in manifest['runner_sha256'].items() if n.endswith('.py') and '/' not in n)
        versions.update({(n,sha):ROOT/'source_versions'/(sha+'_'+n) for n,sha in rec['script_sha256'].items()})
        log=Path(rec['path'])/'run.log';text=log.read_text()
        parsed=r.parse_output(text,rec['kind'],rec.get('expected'),rec.get('repeats',1),rec.get('regression_kind'))
        if rec['kind']=='accuracy':
            assert [list(x) for x in parsed['comparisons']]==rec['comparisons']
            count=rec['shape']['M']*rec['shape']['N']*rec['shape']['G']
            for _,actual,failed,error in parsed['comparisons']:
                assert int(actual)==count and int(failed)==0
                ck_full+=1;ck_elements+=count
        elif rec['kind']=='perf':
            assert parsed['result']==rec['result']
        elif rec['kind']=='native_perf':
            assert parsed['native']==rec['native'];value=rec['native']
            count=rec['shape']['M']*rec['shape']['N']*rec['shape']['G']
            assert value['accuracy_pass'] and value['gpu']==4
            for cmp in value['comparisons']:
                assert cmp['checked']==count and cmp['failed']==0 and cmp['finite'] and cmp['guards_ok']
                native_full+=1;native_elements+=count
        raw.append(dict(key=key,kind=rec['kind'],epoch=rec['epoch'],log_sha256=plan.sha(log)))
    for (name,sha),path in versions.items():assert path.exists() and plan.sha(path)==sha,(name,sha)
    phase_summary=[];all_pairs=[]
    phases=[('H1','mainline',48),('H2','shared',39),('H2 layout','tuning',74),('H3','h3',85),('H3 depth/host','h3_factors',116),('H3 resources','resource',46)]
    for label,stem,total in phases:
        path=ROOT/(stem+'_discovery_results.json');rows=r.read(path);analysis=r.read(ROOT/(stem+'_discovery_analysis.json'))
        assert len(rows)==total and analysis['input_sha256']==plan.sha(path)
        for row in rows:
            rr=[bykey[k] for k in row['keys']];assert len(rr)==8
            assert [x['kind'] for x in rr]==['accuracy']*2+['perf']*6
            assert [x['variant'] for x in rr]==['base','candidate','base','candidate','candidate','base','base','candidate']
            assert [x.get('round') for x in rr[2:]]==[1,1,2,2,3,3]
            assert all(x['epoch']==row['epoch'] and x['benchmark_sha256']==row['binary_sha256'] and x['plan_sha256']==row['plan_sha256'] for x in rr)
            a=[x['result'] for x in rr if x['kind']=='perf' and x['variant']=='base'];b=[x['result'] for x in rr if x['kind']=='perf' and x['variant']=='candidate']
            assert a==row['base_rounds'] and b==row['candidate_rounds']
            am=statistics.median(x['median_ms'] for x in a);bm=statistics.median(x['median_ms'] for x in b)
            assert am==row['base_ms'] and bm==row['candidate_ms']
            assert math.isclose(row['delta_pct'],100*(bm/am-1),abs_tol=1e-9)
            assert row['high_cv']==any(x['cv']>.05 for x in a+b)
            all_pairs.append(dict(phase=label,**row))
        phase_summary.append(dict(phase=label,pairs=total,source_sha256=plan.sha(path),groups=analysis['groups']))
    assert len(all_pairs)==408
    folder=ROOT/'final_attempts/attempt_1';final=r.read(folder/'results.json');data=r.read(folder/'plan.json');original={x['ID']:x for x in r.read(ROOT/'source_results.json')}
    assert len(final)==166
    for row in final:
        rr=[bykey[k] for k in row['keys']];assert len(rr)==16
        assert [x['kind'] for x in rr]==['accuracy']*2+['perf']*14
        assert [x['variant'] for x in rr]==['base','candidate']+sum([['base','candidate'] if n%2 else ['candidate','base'] for n in range(1,8)],[])
        aa=[x['result'] for x in rr if x['kind']=='perf' and x['variant']=='base'];bb=[x['result'] for x in rr if x['kind']=='perf' and x['variant']=='candidate']
        assert aa==row['base_blocks'] and bb==row['candidate_blocks']
        judged=acceptance.classify(original[row['ID']],aa,bb);assert all(row[k]==v for k,v in judged.items())
    report=r.read(folder/'report.json');assert not report['complete'] and not report['goal_performance_pass']
    assert report['stable_weak_originals']==report['attributable_crossings']==0
    packages=[]
    for number in range(1,9):
        entry=r.read(ROOT/f'profile_delivery_batch{number}.json');p=Path(entry['path'])
        assert p.stat().st_size==entry['bytes'] and plan.sha(p)==entry['sha256']
        with tarfile.open(p,'r:gz') as archive:
            members=archive.getmembers();assert all(not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in members)
            readmes=[x for x in members if x.name.endswith('XCOMPUTE_README.md')]
            assert readmes
            assert any(x.name.endswith('profile_manifest.json') for x in members)
        packages.append(dict(number=number,**entry,tar_members=len(members),README_verified=True))
    new=r.read(ROOT/'compile/build9/completed.json');isa=r.read(ROOT/'phase_build_isa_audit.json')
    assert isa['ok'] and isa['old_shader_count']==1710 and new['benchmark_sha256']==plan.sha(plan.BENCH)==isa['benchmark_sha256']
    assert all(plan.sha(plan.REPO/n)==sha for n,sha in new['source_sha256'].items())
    assert not any(x['benchmark_sha256']==new['benchmark_sha256'] for x in jobs)
    assert not (ROOT/'phase_qualification.json').exists() and not (ROOT/'phase_discovery_results.json').exists()
    assert plan.sha(plan.REPO/'docs/data/decode_depth8_replay_p6k/results.json')=='00c80c665735d59165947e5c9458e4e3cb0327df1a47dee2926fe725af4e9d3d'
    for path in ['/workspace/shapes_1178.csv','/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv']:
        assert plan.sha(path)=='5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
    result=dict(at=r.stamp(),ok=True,cpu_only=True,GPU_submitted=False,compiler_invoked=False,
        GPU_jobs=len(jobs),all_jobs_finished=True,pending_started_jobs=0,kinds=dict(collections.Counter(x['kind'] for x in jobs)),epochs=dict(collections.Counter(x['epoch'] for x in jobs)),
        all_GPU4_and_both_visible_flags=True,all_health_kernel_checks_and_raw_outputs_verified=True,
        all_submissions_before_original_cutoff=True,CK_full_comparisons=ck_full,CK_full_elements=ck_elements,
        Native_full_comparisons=native_full,Native_full_elements=native_elements,failed_elements=0,
        unique_original_IDs=len({x['ID'] for x in jobs if isinstance(x.get('ID'),int) and 1<=x['ID']<=1193}),
        full1193_or881_GPU_run=False,source_versions_verified=len(versions),manifests_verified=len(manifests),
        exploratory_pairs=408,exploratory_phases=phase_summary,independent_complete_pairs=166,independent_plan_pairs=347,
        independent_whole_main_rejected=True,adopted_families=[],accepted_stable50=0,accepted_attributable30=0,performance_Goal_complete=False,
        current_binary=new['benchmark_sha256'],current_phase_compiled_only=True,current_phase_GPU_validated=False,
        immutable_P6k_and_shapes_verified=True,perf_packages=packages,original_session=session,
        last_GPU_submission=max(x['at'] for x in jobs),last_GPU_finish=max(x['finished_at'] for x in jobs),
        remaining='Phase27 qualification/46 discovery/profile9 and a new whole-domain+24 unseen seven-block confirmation were not run; fixed submission cutoff has passed. No adopted update to P6k.')
    r.write(OUT/'audit.json',result);r.write(OUT/'raw_log_sha256.json',raw);r.write(OUT/'exploratory_pairs.json',all_pairs)
    print('CLOSEOUT_EXISTING_EVIDENCE_AUDIT_PASS',len(jobs),'jobs',408,'three-round pairs',166,'independent pairs',ck_full,'CK full comparisons',native_full,'Native full comparisons; performanceGoal incomplete',flush=True)

if __name__=='__main__':main()
