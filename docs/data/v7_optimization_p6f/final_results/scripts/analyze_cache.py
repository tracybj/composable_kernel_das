"""Three-round matched comparisons, including all regressions and CVs."""
import csv,json,statistics
from pathlib import Path
import runtime as r
raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
frozen={int(x['ID']):x for x in csv.DictReader(open('/workspace/bf16NN-groupgemm.csv',encoding='utf-8-sig'))}
groups={}
for x in raw:
    if not x.get('ok') or x.get('kind')!='perf' or not x.get('pair','').startswith('cache_'):continue
    groups.setdefault((x['id'],x['variant'],x['epoch']),{})[(x['round'],x['arm'])]=x
rows=[]
for (i,v,ep),data in sorted(groups.items()):
    if not all((rnd,arm) in data for rnd in [1,2,3] for arm in ['control','candidate']):continue
    b=[data[(rnd,'control')] for rnd in [1,2,3]];a=[data[(rnd,'candidate')] for rnd in [1,2,3]]
    before=[x['result']['median_ms'] for x in b];after=[x['result']['median_ms'] for x in a]
    changes=[100*(a/b-1) for a,b in zip(after,before)]
    allrecords=b+a;assert len({x['binary_sha256'] for x in allrecords})==1
    assert len({x['physical_gpu'] for x in allrecords})==1
    cv=[x['result']['cv'] for x in allrecords]
    delta=statistics.median(changes);f=float(frozen[i]['BLAS ms'])
    # Conservative uncertainty: effect must exceed2% and observed round/sample
    # variation. CV is descriptive, not a confidence interval for the median.
    noise=max(2,100*max(cv),100*(max(before)-min(before))/statistics.median(before),100*(max(after)-min(after))/statistics.median(after))
    rows.append(dict(id=i,**{k:int(frozen[i][k]) for k in ['M','N','K','G']},variant=v,epoch=ep,control_instance=b[0]['selected'],candidate_instance=a[0]['selected'],before_ms=statistics.median(before),after_ms=statistics.median(after),paired_delta_pct=delta,observed_variation_band_pct=noise,before_round_ms=before,after_round_ms=after,round_delta_pct=changes,before_cv=[x['result']['cv'] for x in b],after_cv=[x['result']['cv'] for x in a],R_before_frozen=f/statistics.median(before),R_after_frozen=f/statistics.median(after),frozen_BLAS_ms=f,frozen_BLAS_gpu=0,CK_gpu=allrecords[0]['physical_gpu'],quality='uncertain_within2pct' if abs(delta)<=2 else 'within_observed_variation_uncertain' if abs(delta)<=noise else 'stable_direction' if all(x*delta>0 for x in changes) else 'mixed_rounds',binary_sha256=allrecords[0]['binary_sha256'],source_keys=[x['key'] for x in allrecords],source_logs=[str(Path(x['path'])/'run.log') for x in allrecords]))
# Select latest *complete* epoch; never mix arms/rounds from different epochs.
latest={}
for x in rows:
    key=(x['id'],x['variant'])
    if key not in latest or x['epoch']>latest[key]['epoch']:latest[key]=x
rows=sorted(latest.values(),key=lambda x:(x['id'],x['variant']))
summary=[]
for v in sorted({x['variant'] for x in rows}):
    xs=[x for x in rows if x['variant']==v]
    if len(xs)!=4:continue
    paired_totals=[100*(sum(x['after_round_ms'][n] for x in xs)/sum(x['before_round_ms'][n] for x in xs)-1) for n in range(3)]
    summary.append(dict(variant=v,shapes=4,median_shape_paired_delta_pct=statistics.median(x['paired_delta_pct'] for x in xs),paired_total_delta_pct=statistics.median(paired_totals),paired_total_round_pct=paired_totals,gains_over2pct=sum(x['paired_delta_pct']< -2 for x in xs),regressions_over2pct=sum(x['paired_delta_pct']>2 for x in xs),uncertain_count=sum(x['quality']!='stable_direction' for x in xs),note='Four predeclared shapes, each invoked once for total; no production frequency weighting'))
(r.ROOT/'cache_comparisons.json').write_text(json.dumps(rows,indent=2));(r.ROOT/'cache_factor_summary.json').write_text(json.dumps(summary,indent=2))
if rows:
    with (r.ROOT/'cache_comparisons.csv').open('w') as f:
        fields=[k for k,v in rows[0].items() if not isinstance(v,list)];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('CACHE_COMPLETE_COMPARISONS',len(rows))
for x in summary:print('FACTOR',x['variant'],'median',round(x['median_shape_paired_delta_pct'],2),'total',round(x['paired_total_delta_pct'],2),'regressions',x['regressions_over2pct'])
