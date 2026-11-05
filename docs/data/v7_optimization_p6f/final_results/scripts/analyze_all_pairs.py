"""Same-binary, same-GPU, same-epoch complete three-round comparisons only."""
import csv,json,re,statistics
from pathlib import Path
import runtime as r

best={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv'))}
raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
exclusion_file=r.ROOT/'measurement_exclusions.json'
excluded=set(json.loads(exclusion_file.read_text())['keys']) if exclusion_file.exists() else set()
contracts={
    5013:'Tail128: common nonragged/nonempty gfx936 BF16NN, M<=128,N>=128,N%8=0,N%128!=0,K64..4096%16,G<=1024; signed per-group A/B byte spans and total tile bounds',
    5016:'Tail256: common nonragged/nonempty gfx936 BF16NN, M<=128,N>=128,N%8=0,N%128!=0,K64..4096%16,G<=1024; signed per-group A/B byte spans and total tile bounds',
    5014:'Small-M mode5 K32 steady: common nonragged/nonempty gfx936 BF16NN,M<=64,N%128=0,K128..4096%32,G<=1024; signed per-group B byte span and total tile bounds',
    5015:'Small-M mode6 K16 steady: common nonragged/nonempty gfx936 BF16NN,M<=64,N%128=0,K64..4096%16,G<=1024; signed per-group B byte span and total tile bounds',
    5017:'Small-M mode7 NoSLC: common nonragged/nonempty gfx936 BF16NN,M<=64,N%128=0,K64..4096%16,G<=1024; signed per-group B byte span and total tile bounds',
}
selector_source=r.REPO/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'
groups={}
for x in raw:
    if x.get('ok') and x.get('kind')=='perf' and x['key'] not in excluded:
        study=re.sub(r'_r[123](?=_|$)','',x['pair'])
        group=(x['id'],x.get('variant','default'),x['epoch'],x['binary_sha256'],study)
        position=(x['round'],x['arm'])
        existing=groups.setdefault(group,{}).get(position)
        if existing is None or x['at']>existing['at']:groups[group][position]=x
rows=[];partial=[]
for (i,variant,epoch,binary,study),jobs in sorted(groups.items()):
    required={(rd,arm) for rd in (1,2,3) for arm in ('control','candidate')}
    if not required<=set(jobs):
        partial.append(dict(id=i,variant=variant,epoch=epoch,binary_sha256=binary,study=study,positions=sorted(jobs)));continue
    b=[jobs[(rd,'control')] for rd in (1,2,3)];a=[jobs[(rd,'candidate')] for rd in (1,2,3)]
    alljobs=b+a;assert len({x['physical_gpu'] for x in alljobs})==1
    assert all(not x['kernel_service_events'] for x in alljobs)
    before=[x['result']['median_ms'] for x in b];after=[x['result']['median_ms'] for x in a]
    changes=[100*(c/t-1) for c,t in zip(after,before)]
    med=statistics.median;delta=med(changes)
    cv=[x['result']['cv'] for x in alljobs]
    variation=max(2,100*max(cv),100*(max(before)-min(before))/med(before),100*(max(after)-min(after))/med(after))
    ref=best[i];shape=a[0]['shape'];blas=float(ref['BLAS_ms'])
    quality=('uncertain_within2pct' if abs(delta)<=2 else 'within_observed_variation_uncertain' if abs(delta)<=variation
             else 'stable_direction' if all(x*delta>0 for x in changes) else 'mixed_rounds')
    precision=[x['key'] for x in raw if x.get('ok') and x.get('kind')=='accuracy' and x.get('id')==i
               and x.get('selected')==a[0]['selected'] and x['binary_sha256']==binary]
    assert all(a0['pair']==b0['pair'] for a0,b0 in zip(a,b)),'Arms from different adjacent contrasts'
    rows.append(dict(id=i,**{k:shape[k] for k in ['M','N','K','G']},task=a[0].get('task',2),variant=variant,study=study,
        control_instance=b[0]['selected'],candidate_instance=a[0]['selected'],
        control_scheme=b[0].get('baseline_scheme','explicit matched geometry/control'),
        control_env={k:v for k,v in b[0]['environment'].items() if k.startswith('CK_TILE_GROUPED_GEMM_')},
        candidate_env={k:v for k,v in a[0]['environment'].items() if k.startswith('CK_TILE_GROUPED_GEMM_')},
        before_ms=med(before),after_ms=med(after),before_round_ms=before,after_round_ms=after,
        paired_delta_pct=delta,round_delta_pct=changes,before_cv=[x['result']['cv'] for x in b],after_cv=[x['result']['cv'] for x in a],
        observed_variation_band_pct=variation,quality=quality,
        frozen_BLAS_ms=blas,frozen_BLAS_gpu=int(ref['BLAS_gpu']),frozen_BLAS_protocol=ref['BLAS_protocol'],
        frozen_BLAS_source=('/workspace/bf16NN-groupgemm.csv' if i<1179 else '/workspace/v7_coverage/p6c/new_blas/run_config.json'),
        R_before_frozen=blas/med(before),R_after_frozen=blas/med(after),
        CK_gpu=alljobs[0]['physical_gpu'],epoch=epoch,binary_sha256=binary,last_measured_at=max(x['finished_at'] for x in alljobs),
        source_keys=[x['key'] for x in alljobs],source_logs=[str(Path(x['path'])/'run.log') for x in alljobs],
        full_output_accuracy_keys=precision,fallback=a[0].get('fallback',False),
        selection_contract=(contracts.get(a[0]['selected']) or
            ('Bounded CACHE_GRID: common nonragged/nonempty gfx936 BF16NN,1024<M<=2048,N>4096%256,K3072..4096%16,G<=1024; signed A/B byte spans and total tile bounds' if variant=='cache_policy' and not a[0].get('fallback',False) else
             'Actual fallback/original/diagnostic route determined by complete effective policy flags and MNKG; see exact source and frozen plan; no forced-instance CLI')),
        selector_source=str(selector_source),selector_source_sha256=r.sha(selector_source)))
# Keep every completed generation, while providing an explicit latest-complete view.
(r.ROOT/'all_pair_generations.json').write_text(json.dumps(rows,indent=2))
latest={}
for x in rows:
    key=(x['id'],x['variant'])
    if key not in latest or x['last_measured_at']>latest[key]['last_measured_at']:latest[key]=x
rows=sorted(latest.values(),key=lambda x:(x['task'],x['variant'],x['id']))
(r.ROOT/'per_shape_results.json').write_text(json.dumps(rows,indent=2))
(r.ROOT/'partial_pair_groups.json').write_text(json.dumps(partial,indent=2))
if rows:
    with (r.ROOT/'per_shape_results.csv').open('w',newline='') as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for x in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in x.items()})
for variant in sorted({x['variant'] for x in rows}):
    xs=[x for x in rows if x['variant']==variant]
    total=med(100*(sum(x['after_round_ms'][rd] for x in xs)/sum(x['before_round_ms'][rd] for x in xs)-1) for rd in range(3))
    print('PAIRS',variant,'shapes',len(xs),'median',round(med(x['paired_delta_pct'] for x in xs),3),'total',round(total,3),
          'regressions>2pct',sum(x['paired_delta_pct']>2 for x in xs),'uncertain',sum(x['quality']!='stable_direction' for x in xs))
print('COMPLETE',len(rows),'PARTIAL',len(partial))
