"""Audit complete groups and assess predeclared property ranges."""
import csv,datetime,json,math,re,statistics
from pathlib import Path
import runtime as r
import plan

ROOT=Path(__file__).resolve().parent

def main():
    frozen=json.loads((ROOT/'scope_plan.json').read_text())
    groups=[json.loads(x) for x in (ROOT/'shape_results.jsonl').read_text().splitlines()]
    jobs=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
    lookup={x['key']:x for x in jobs};assert len(lookup)==len(jobs)
    gates={x['key']:x for x in map(json.loads,(ROOT/'gate.jsonl').read_text().splitlines())}
    keys_used=set();rows=[];comparisons=elements=0
    for group in groups:
        assert group['binary_sha256']==frozen['binary_sha256']
        recs=[lookup[k] for k in group['keys']];keys_used.update(group['keys'])
        acc={x['variant']:x for x in recs if x['kind']=='accuracy'}
        assert len(acc)==len(group['instances'])
        for job in recs:
            assert job['ok'] and not job['kernel_service_events'] and job['epoch']==group['epoch']
            assert job['binary_sha256']==group['binary_sha256'] and job['physical_gpu']==4
            assert job['selected']==job['expected']==group['instances'][job['variant']]
            assert job['environment']['HIP_VISIBLE_DEVICES']==job['environment']['CUDA_VISIBLE_DEVICES']=='4'
            assert job['command'][0]=='/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark'
            assert '--include-args-build=0' in job['command'] and '--launches-per-sample=1' in job['command']
            gate=gates[job['key']]
            assert gate['rc']==0 and gate['hcu']=='0.0%' and 'MKFD' not in gate['output'].upper()
            if job['kind']=='accuracy':
                size=math.prod(group['shape'][k] for k in ['M','N','G'])
                assert len(job['comparisons'])==3
                for repeat,checked,failed,max_abs in job['comparisons']:
                    assert int(checked)==size and int(failed)==0 and float(max_abs)==0
                    comparisons+=1;elements+=size
                text=Path(job['path'],'run.log').read_text()
                for backend in ['ck_redzone','blas_redzone']:
                    guards=re.findall('ACCURACY backend='+backend+r' repeat=\d+ checked=(\d+) failed=(\d+)',text)
                    assert len(guards)==3 and all(int(n)==32 and int(f)==0 for n,f in guards)
        if not group['timed']:continue
        first_perf=min(datetime.datetime.fromisoformat(x['at']) for x in recs if x['kind']=='perf')
        assert all(datetime.datetime.fromisoformat(x['finished_at'])<first_perf for x in acc.values())
        for arm in group['instances']:
            measured=[x for x in recs if x['kind']=='perf' and x['variant']==arm]
            assert len(measured)==3
            assert [x['result']['median_ms'] for x in measured]==group['times'][arm]
        reference=min((a for a in group['median_ms'] if a!='new'),key=lambda a:group['median_ms'][a])
        base=group['median_ms'][reference];candidate=group['median_ms']['new']
        changes=[(n/b-1)*100 for n,b in zip(group['times']['new'],group['times'][reference])]
        row=dict(ID=group['id'],suite=group['suite'],**group['shape'],baseline_arm=reference,
          base_instance=group['instances'][reference],candidate_instance=group['instances']['new'],
          before_ms=base,after_ms=candidate,change_pct=(candidate/base-1)*100,
          three_round_improved_gt2=all(x < -2 for x in changes),three_round_regressed_gt2=all(x>2 for x in changes),
          high_cv=max(group['cvs']['new']+group['cvs'][reference])>.05,
          round_changes_pct=changes,round_ms=group['times'],round_cv=group['cvs'],
          binary_sha256=group['binary_sha256'],physical_gpu=4,epoch=group['epoch'],keys=group['keys'])
        rows.append(row)
    def summarize(rs):
        return dict(cases=len(rs),improved_gt2=sum(x['change_pct'] < -2 for x in rs),
          regressed_gt2=sum(x['change_pct'] > 2 for x in rs),within2=sum(abs(x['change_pct'])<=2 for x in rs),
          three_round_improved_gt2=sum(x['three_round_improved_gt2'] for x in rs),
          high_cv=sum(x['high_cv'] for x in rs),
          median_change_pct=statistics.median(x['change_pct'] for x in rs) if rs else None,
          sum_change_pct=(sum(x['after_ms'] for x in rs)/sum(x['before_ms'] for x in rs)-1)*100 if rs else None,
          worst_change_pct=max((x['change_pct'] for x in rs),default=None))
    discovery=[x for x in rows if x['suite'] in ['existing','unseen']]
    regions={}
    for name in plan.REGIONS:
        members=[x for x in discovery if plan.region(x,name)]
        existing=[x for x in members if x['suite']=='existing'];unseen=[x for x in members if x['suite']=='unseen']
        expected=sum(plan.region(x,name) for x in frozen['existing']+frozen['unseen_probes'])
        stats=summarize(members)
        eligible=len(members)==expected and len(unseen)>=4 and all(x['three_round_improved_gt2'] for x in members) and stats['median_change_pct']<=-3
        regions[name]=dict(predicate=frozen['property_predicates'][name],expected=expected,observed=len(members),
          existing=summarize(existing),unseen=summarize(unseen),combined=stats,
          eligible_for_confirmation=eligible,member_ids=[x['ID'] for x in members])
    done={suite:sum(x['suite']==suite for x in groups) for suite in ['existing','unseen','boundary','confirmation']}
    raw_unaccepted=[x['key'] for x in jobs if x['key'] not in keys_used]
    checks=[json.loads(x) for x in (ROOT/'kernel_checks.jsonl').read_text().splitlines()]
    assert all(x['rc']==0 for x in checks)
    excluded_path=ROOT/'known_excluded_jobs.json'
    excluded=json.loads(excluded_path.read_text()) if excluded_path.exists() else []
    excluded_keys={x['key'] for x in excluded}
    assert not (excluded_keys & keys_used)
    summary=dict(ok=True,binary_sha256=frozen['binary_sha256'],completed_groups=done,
      discovery_complete=done['existing']==80 and done['unseen']==24 and done['boundary']==4,
      all_discovery=summarize(discovery),existing=summarize([x for x in rows if x['suite']=='existing']),
      unseen=summarize([x for x in rows if x['suite']=='unseen']),
      regions=regions,full_output_comparisons=comparisons,full_output_elements=elements,
      max_abs=0,raw_jobs=len(jobs),failed_jobs=[x['key'] for x in jobs if not x['ok']],
      unresolved_failed_jobs=[x['key'] for x in jobs if not x['ok'] and x['key'] not in excluded_keys],
      excluded_raw_jobs=excluded,
      incomplete_or_excluded_raw_keys=raw_unaccepted,physical_gpu=4,
      unresolved_incomplete_raw_keys=[x for x in raw_unaccepted if x not in excluded_keys],
      interpretation='Finite discovery and unseen-shape probes, not full1193 or mathematical proof for all future shapes; comparison uses fresh same-binary best available declared controls')
    (ROOT/'results.json').write_text(json.dumps(rows,indent=2))
    with (ROOT/'results.csv').open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
            for x in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in x.items()})
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('SCOPE_DATA_AUDIT_PASS groups='+str(done),'existing='+str(summary['existing']),flush=True)
    for name,x in regions.items():
        print('REGION_STATUS',name,x['observed'],'/',x['expected'],x['combined'],'confirm='+str(x['eligible_for_confirmation']),flush=True)

if __name__=='__main__':main()
