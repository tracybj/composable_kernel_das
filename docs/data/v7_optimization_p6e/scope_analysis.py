#!/usr/bin/env python3
"""Publish every contrast, regression and attribute subgroup, without an oracle."""
import collections,csv,json,statistics
from pathlib import Path
import analyze_all as analysis

p=analysis.P

def ctas(row,tm,tn):return ((row['M']+tm-1)//tm)*((row['N']+tn-1)//tn)*row['G']

def cta_bucket(count):return '<80' if count<80 else '80..319' if count<320 else '320..639' if count<640 else '>=640'

def write(name,rows):
    (p/(name+'.json')).write_text(json.dumps(rows,indent=2))
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (p/(name+'.csv')).open('w') as output:
            writer=csv.DictWriter(output,fields)
            writer.writeheader()
            writer.writerows(rows)

def main():
    expanded=json.loads((p/'small_expanded.json').read_text())
    assert len(expanded)==310,'Finish all predeclared scopes first'
    groups=[]
    for name in sorted({row['strategy'] for row in expanded}):
        rows=[row for row in expanded if row['strategy']==name]
        for dimension,keyfn in [
            ('all',lambda row:'all'),
            ('M',lambda row:str(row['M'])),
            ('K',lambda row:str(row['K'])),
            ('CTA',lambda row:cta_bucket(ctas(row,64,128))),
            ('original_CTA',lambda row:cta_bucket(ctas(row,256,256)))]:
            buckets=collections.defaultdict(list)
            for row in rows:buckets[keyfn(row)].append(row)
            for label,group in sorted(buckets.items()):
                stats=analysis.aggregate(group)
                scalar={key:value for key,value in stats.items() if not isinstance(value,list)}
                groups.append(dict(strategy=name,dimension=dimension,bucket=label,cta_definition='ceil(M/64)*ceil(N/128)*G; common reference geometry',original_delta_median_pct=statistics.median(row['after_vs_original_pct'] for row in group),original_regress_gt2=sum(row['after_vs_original_pct']>2 for row in group),**scalar))
    write('scope_subgroups',groups)
    rows=[]
    for source in ['cache_expanded','small_representatives','small_expanded']:
        for row in json.loads((p/(source+'.json')).read_text()):
            original=row.get('original_ms',row['before_ms'])
            original_source=row.get('original_source','current card4 original route, three rounds' if source=='cache_expanded' else 'current card4 original5000; separately retained representative arm')
            rows.append(dict(source=source+'.json',strategy=row.get('strategy',row.get('variant')),id=row['id'],M=row['M'],N=row['N'],K=row['K'],G=row['G'],baseline_variant=row.get('baseline_variant','original5002' if row['K']>4096 else 'original5000'),variant=row['variant'],original_ck_ms=original,original_source=original_source,baseline_ck_ms=row['before_ms'],new_ck_ms=row['after_ms'],paired_delta_vs_baseline_pct=row['delta_pct'],delta_vs_original_pct=row.get('after_vs_original_pct',row['delta_pct']),original_comparison_note='paired three-round median' if source in ['small_expanded','cache_expanded'] else 'descriptive ratio of medians; see separate original arm',regress_vs_baseline_gt2=row['delta_pct']>2,regress_vs_original_gt2=row.get('after_vs_original_pct',row['delta_pct'])>2,observational_label=row['observational_label'],blas_ms=row['blas_ms'],blas_gpu=row['blas_gpu'],blas_protocol=row['blas_protocol'],blas_source=row['blas_source'],ck_gpu=4,ck_protocol='warmup10 samples30 single launch, include_args_build0',R_frozen_after=row['R_after'],environment_epochs=json.dumps(row['environment_epochs']),commands=json.dumps(row['commands'])))
            tm,tn={'drain':(64,128),'k32':(64,128),'square':(128,128),'k32n64':(64,64),'noslc':(256,256)}[row['variant']]
            rows[-1].update(original_ctas=ctas(row,256,256),new_ctas=ctas(row,tm,tn),new_tile_m=tm,new_tile_n=tn)
    write('per_shape_results',rows)
    write('all_regressions',[row for row in rows if row['regress_vs_baseline_gt2'] or row['regress_vs_original_gt2']])
    print('SCOPE_ANALYSIS_PASS contrasts='+str(len(rows))+' subgroups='+str(len(groups)))

if __name__=='__main__':main()
