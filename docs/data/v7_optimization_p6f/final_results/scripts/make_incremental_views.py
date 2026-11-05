"""Source-labelled1193 mosaics; neither view is a fresh uniform GPU replay."""
import csv,json,math,statistics
from pathlib import Path
import runtime as r

p=r.ROOT
original=list(csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv')))
assert len(original)==1193
base={int(x['ID']):x for x in original}
series={x['candidate_id']:x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/all_candidate_series.csv'))}
old_full={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_known_1193.csv'))}
comparisons=json.loads((p/'all_pair_generations.json').read_text())
latest=json.loads((p/'per_shape_results.json').read_text())
raw=[json.loads(x) for x in (p/'jobs.jsonl').read_text().splitlines()]
good={x['key']:x for x in raw if x.get('ok')}

def historical(x):
    i=int(x['ID']);ms=float(x['best_ms'])
    old=series[x['candidate_id']];times=json.loads(old['round_median_ms'])
    if times:assert math.isclose(statistics.median(times),ms,rel_tol=1e-9,abs_tol=1e-12)
    digests=json.loads(old['binary_sha256'])
    known=[v for v in digests if isinstance(v,str) and len(v)==64]
    return dict(ID=i,**{k:int(x[k]) for k in ['M','N','K','G']},phase=x['phase'],
        scheme=x['best_scheme'],instance_id=int(x['best_instance_id']),CK_ms=ms,
        frozen_BLAS_ms=float(x['BLAS_ms']),R_frozen=float(x['BLAS_ms'])/ms,
        CK_gpu=4,BLAS_gpu=int(x['BLAS_gpu']),BLAS_protocol=x['BLAS_protocol'],
        BLAS_source=old_full[i]['BLAS_source'],BLAS_high_span_warning=old_full[i]['BLAS_high_span_warning'],
        CK_protocol='single_launch_warmup10_samples30_include_args_build0',
        enable_env=json.loads(x['enable_env']),source_type='historical_current_retained',
        source=x['source'],source_keys=json.loads(x['source_keys']),source_logs=[],
        binary_sha256=known[0] if len(known)==1 else json.dumps(known) if known else '',
        epoch=None,round_ms=times,round_cv=[],max_sample_cv=old['max_sample_cv'],
        quality=x['quality'],full_output_accuracy_keys=[],
        original_scheme=x['best_scheme'],original_CK_ms=ms,original_R=float(x['R_frozen']),
        note='Historical source retained; no new1193-shape GPU run or unified per-shape dispatch policy')

def measured(pair,arm):
    keys=[k for k in pair['source_keys'] if good[k]['arm']==arm]
    jobs=sorted((good[k] for k in keys),key=lambda x:x['round'])
    assert len(jobs)==3 and {x['round'] for x in jobs}=={1,2,3}
    assert len({x['epoch'] for x in jobs})==len({x['binary_sha256'] for x in jobs})==1
    assert all(x['kind']=='perf' and x['physical_gpu']==4 for x in jobs)
    assert all(not any(a.startswith('--launches-per-sample=') and a!='--launches-per-sample=1' for a in x['command']) for x in jobs)
    i=pair['id'];out=historical(base[i]);times=[x['result']['median_ms'] for x in jobs]
    precision=[x['key'] for x in raw if x.get('ok') and x.get('kind')=='accuracy' and x.get('id')==i and x.get('selected')==jobs[0]['selected'] and x['binary_sha256']==jobs[0]['binary_sha256']]
    ms=statistics.median(times)
    out.update(scheme='P6f_'+pair['variant']+'_'+arm,instance_id=jobs[0]['selected'],CK_ms=ms,
        R_frozen=out['frozen_BLAS_ms']/ms,
        enable_env={k:v for k,v in jobs[0]['environment'].items() if k.startswith('CK_TILE_GROUPED_GEMM_')},
        source_type='p6f_fresh_three_round_series',source=str(p/'jobs.jsonl'),
        source_keys=[x['key'] for x in jobs],source_logs=[str(Path(x['path'])/'run.log') for x in jobs],
        binary_sha256=jobs[0]['binary_sha256'],epoch=jobs[0]['epoch'],round_ms=times,
        round_cv=[x['result']['cv'] for x in jobs],max_sample_cv=max(x['result']['cv'] for x in jobs),quality=pair['quality'],full_output_accuracy_keys=precision,
        note='Three round medians; pair quality describes its causal contrast, not proof of globally fastest scheme')
    return out

policy={i:historical(x) for i,x in base.items()}
policy_rows=[x for x in latest if x['variant']=='tail256' or (x['variant']=='cache_policy' and not x['fallback'])]
assert len(policy_rows)==24
for x in policy_rows:
    assert x['quality']=='stable_direction' and x['paired_delta_pct']< -2
    row=measured(x,'candidate');row['source_type']='p6f_adopted_attribute_policy_measurement'
    row['note']='Verified default-off attribute route overlaid on historical per-shape baseline mosaic; not a uniform1193 replay'
    policy[x['id']]=row

envelope={i:historical(x) for i,x in base.items()}
for pair in comparisons:
    for arm in ['control','candidate']:
        row=measured(pair,arm);i=row['ID']
        if row['CK_ms']<envelope[i]['CK_ms']:
            row['note']='Numerical minimum of historical baseline and complete current-code P6f series; not a deployed oracle or stable-optimality proof'
            envelope[i]=row

def bins(rows):
    totals={}
    for phase in ['prefill','decode','all']:
        xs=[x for x in rows if phase=='all' or x['phase']==phase];counts=[0]*6
        for x in xs:
            v=x['R_frozen'];assert math.isfinite(v) and v>0
            index=0 if v>=.99 else 1 if v>=.8 else 2 if v>=.5 else 3 if v>=.3 else 4 if v>=.2 else 5
            counts[index]+=1
        assert sum(counts)==len(xs)
        totals[phase]=dict(total=len(xs),bins=counts,fresh=sum(x['source_type']!='historical_current_retained' for x in xs))
    assert [totals[k]['total'] for k in ['prefill','decode','all']]==[312,881,1193]
    return totals

views=[('policy_overlay_incremental_1193',policy),('historical_envelope_incremental_1193',envelope)]
summary=dict(at=r.stamp(),baseline=bins([historical(x) for x in original]),
    bins=['>=.99','[.8,.99)','[.5,.8)','[.3,.5)','[.2,.3)','<.2'],views={},
    scope='Both1193 tables retain historical source labels. Only affected domains are fresh; never claim fresh full-GPU replay. Frozen Native remains old1178GPU0/repeat20,new15GPU4/repeat20.',
    adopted_rules=['N-tail:common nonragged/nonempty gfx936 BF16NN,M<=128,N>=128,N%8=0,N%128!=0,K64..4096%16,G<=1024 plus byte/tile bounds; choose tail256',
                   'Cache:common nonragged/nonempty gfx936 BF16NN,1024<M<=2048,N>4096,N%256=0,K3072..4096%16,G<=1024 plus byte/tile bounds; NoSLC/M01=5'],
    task3_policy='New K16/K32 flows and complete159-case existing-K32 resource domain do not justify broad adoption; all negative/noise results retained; numerical envelope entries are separately reproducible, not a new global policy')
for name,data in views:
    rows=[data[i] for i in range(1,1194)];assert len(rows)==1193
    (p/(name+'.json')).write_text(json.dumps(rows,indent=2))
    with (p/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    summary['views'][name]=bins(rows)
(p/'incremental_summary.json').write_text(json.dumps(summary,indent=2))
print('INCREMENTAL1193_SOURCE_LABELLED_VIEWS')
for name,data in summary['views'].items():
    print(name)
    for phase,values in data.items():print(phase,values)
