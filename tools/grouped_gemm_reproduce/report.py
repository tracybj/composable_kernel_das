"""Derive only complete accuracy-plus-three-round shape attempts."""
import csv
import json
import math
import statistics
from pathlib import Path
import runtime as r


def records(root):
    p = Path(root)/'jobs.jsonl'
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def completed(root, manifest):
    jobs = records(root); groups = {}; result = {}; partial = []
    cases = {x['ID']: x for x in manifest['table']['cases']}
    for x in jobs:
        if x.get('kind') not in ['accuracy','perf'] or not x.get('ok'): continue
        if x['plan_sha256'] != manifest['plan_sha256'] or x['benchmark_sha256'] != manifest['benchmark_sha256']:
            raise ValueError('Mixed plans or binaries in output directory')
        groups.setdefault((x['ID'], x['epoch'], x['attempt']), {})[(x['kind'],x.get('round',0))] = x
    for (i, epoch, attempt), data in groups.items():
        required = [('accuracy',0), ('perf',1), ('perf',2), ('perf',3)]
        if not all(x in data for x in required):
            partial.append(dict(ID=i,epoch=epoch,attempt=attempt,positions=list(data))); continue
        acc = data[('accuracy',0)]; perf = [data[('perf',rd)] for rd in [1,2,3]]; alljobs=[acc]+perf
        case = cases[i]
        assert len({x['binary_sha256'] for x in alljobs}) == 1
        assert {x['physical_gpu'] for x in alljobs} == {4}
        assert {x['selected'] for x in alljobs} == {case['expected']}
        assert all(x['environment'] == acc['environment'] for x in alljobs)
        assert acc['finished_at'] <= perf[0]['at']
        assert all(perf[n]['finished_at'] <= perf[n+1]['at'] for n in [0,1])
        assert all(not x['kernel_service_events'] for x in alljobs)
        times = [x['result']['median_ms'] for x in perf]; cvs = [x['result']['cv'] for x in perf]
        ms = statistics.median(times); delta = 100*(ms/case['historical_CK_ms']-1)
        band = max(2,100*max(cvs),100*(max(times)-min(times))/ms)
        row = dict(**case, actual_instance=case['expected'], CK_ms=ms, R_frozen=case['frozen_BLAS_ms']/ms,
                   historical_delta_pct=delta, quality='uncertain' if abs(delta)<=band else 'different_from_historical_observation',
                   observed_variation_band_pct=band, round_median_ms=times,
                   round_min_ms=[x['result']['min_ms'] for x in perf], round_max_ms=[x['result']['max_ms'] for x in perf], round_cv=cvs,
                   physical_gpu=4, epoch=epoch, attempt=attempt, binary_sha256=acc['binary_sha256'],
                   plan_sha256=manifest['plan_sha256'], full_output_key=acc['key'], comparisons=acc['comparisons'],
                   redzone_elements_each_backend=32, source_keys=[x['key'] for x in alljobs],
                   source_logs=[str(Path(x['path'])/'run.log') for x in alljobs],
                   measured_at=perf[-1]['finished_at'], source_type='p6g_fresh_full_output_and_three_rounds')
        # Latest complete chronological attempt, never fastest attempt.
        if i not in result or row['measured_at'] > result[i]['measured_at']: result[i] = row
    return result, partial


def bins(rows, historical=False):
    out={}
    for phase in ['prefill','decode','all']:
        xs=[x for x in rows if phase=='all' or x['phase']==phase]; counts=[0]*6
        for x in xs:
            ratio=x['historical_R'] if historical else x['R_frozen']
            assert math.isfinite(ratio) and ratio>0
            j=0 if ratio>=.99 else 1 if ratio>=.8 else 2 if ratio>=.5 else 3 if ratio>=.3 else 4 if ratio>=.2 else 5
            counts[j]+=1
        out[phase]=dict(total=len(xs),bins=counts)
    return out


def emit(root):
    root=Path(root); manifest=r.read(root/'frozen_plan.json')
    done, partial=completed(root,manifest); rows=[done[i] for i in sorted(done)]
    r.write(root/'results.json',rows)
    if rows:
        with (root/'results.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader()
            for x in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in x.items()})
    raw=records(root)
    invalid=[x for x in raw if not x.get('ok')]
    missing=[x['ID'] for x in manifest['table']['cases'] if x['ID'] not in done]
    summary=dict(at=r.stamp(),complete=len(rows)==1193,accepted_shapes=len(rows),expected_shapes=1193,
                 incomplete_ids=missing,partial_attempts=partial,invalid_keys=[x['key'] for x in invalid],
                 new_bins=bins(rows),historical_bins=bins(manifest['table']['cases'],True),
                 expected_totals=[312,881,1193],bin_labels=['>=.99','[.8,.99)','[.5,.8)','[.3,.5)','[.2,.3)','<.2'],
                 recovery=r.read(root/'recovery_total.json',dict(seconds=0,attempts=0)),
                 note='Only complete same-epoch attempts are fresh; missing rows are never filled from history. Frozen Native references unchanged.')
    r.write(root/'summary.json',summary)
    r.write(root/'all_failures.json',invalid)
    r.write(root/'all_regressions.json',[x for x in rows if x['historical_delta_pct']>2])
    crossing=[]
    def bucket(v):return 0 if v>=.99 else 1 if v>=.8 else 2 if v>=.5 else 3 if v>=.3 else 4 if v>=.2 else 5
    for row in rows:
        before,after=bucket(row['historical_R']),bucket(row['R_frozen'])
        if before!=after:crossing.append(dict(ID=row['ID'],before_bin=before,after_bin=after,quality=row['quality'],historical_R=row['historical_R'],fresh_R=row['R_frozen']))
    r.write(root/'bin_crossings.json',crossing)
    return summary
