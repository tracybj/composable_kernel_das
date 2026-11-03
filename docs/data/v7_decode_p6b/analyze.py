#!/usr/bin/env python3
"""Aggregate independent A/B rounds and verify profile geometry."""
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parent/'stage_wait'
sys.path.insert(0,'/workspace/v7_coverage/p4a')
from perfdump import dispatches
spec=importlib.util.spec_from_file_location('p6metrics','/workspace/v7_coverage/p6/analyze.py')
p6=importlib.util.module_from_spec(spec);spec.loader.exec_module(p6)


def main():
    jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
    good={r['key']:r for r in jobs if r.get('ok')}
    with open('/workspace/Grouped_Geem/blaslt_tuning_tools/results/bf16_nn_compare_20261027_191315/groupgemm/results.csv',encoding='utf-8-sig') as f:
        baseline={int(r['ID']):r for r in csv.DictReader(f)}
    perf=[]
    ids=json.loads((ROOT/'perf_ids.json').read_text()) if (ROOT/'perf_ids.json').exists() else []
    for i in ids:
        for tile in [128,64]:
            b=[good[f'perf_r{r}_id{i}_n{tile}_before']['result'] for r in [1,2,3] if f'perf_r{r}_id{i}_n{tile}_before' in good]
            a=[good[f'perf_r{r}_id{i}_n{tile}_after']['result'] for r in [1,2,3] if f'perf_r{r}_id{i}_n{tile}_after' in good]
            if len(b)!=3 or len(a)!=3:continue
            before=statistics.median(r['median_ms'] for r in b);after=statistics.median(r['median_ms'] for r in a)
            row=dict(id=i,tile_n=tile,**{k:int(baseline[i][k]) for k in ['M','N','K','G']},
                     before_ms=before,after_ms=after,delta_pct=100*(after/before-1),speedup=before/after,
                     blas_card0_ms=float(baseline[i]['BLAS ms']),R=float(baseline[i]['BLAS ms'])/after,
                     before_round_ms=[r['median_ms'] for r in b],after_round_ms=[r['median_ms'] for r in a],
                     before_cv=[r['cv'] for r in b],after_cv=[r['cv'] for r in a])
            perf.append(row)
    guard=[]
    for i in [76,358,962,1118,1095,139]:
        b=[good[f'guard_r{r}_id{i}_before']['result']['median_ms'] for r in [1,2,3] if f'guard_r{r}_id{i}_before' in good]
        a=[good[f'guard_r{r}_id{i}_after']['result']['median_ms'] for r in [1,2,3] if f'guard_r{r}_id{i}_after' in good]
        if len(a)==len(b)==3:
            before,after=statistics.median(b),statistics.median(a)
            guard.append(dict(id=i,before_ms=before,after_ms=after,delta_pct=100*(after/before-1),before_round_ms=b,after_round_ms=a))
    counters=[];raw=[]
    for key,r in good.items():
        match=re.fullmatch(r'prof_id(\d+)_n(\d+)',key)
        if not match:continue
        i,tile=map(int,match.groups());shape=tuple(int(baseline[i][k]) for k in ['M','N','K','G'])
        geometry=(64,tile,16) if tile else (256,256,16)
        ds=dispatches(r['perf_files'][0]);assert len(ds)==2
        metric=p6.metrics(ds[-1],shape,geometry)
        expected_waves=metric['ctas']*(4 if tile else 8)
        for d in ds:
            assert d['SQ_INSTS_MMOP']==metric['mmop_insts_expected_padded'],key
            assert d['SQ_WAVES']==expected_waves,(key,d['SQ_WAVES'],expected_waves)
        text=(Path(r['path'])/'run.log').read_text()
        timing=[int(x)/1e6 for x in re.findall(r'dispatch kernel .*?execution\((\d+) ns\)',text)]
        counters.append(dict(id=i,tile_n=tile,**metric,aql_dispatch_ms=timing))
        raw.append(dict(key=key,dispatches=ds,source=r['perf_files'][0]))
    for name,value in [('perf_summary.json',perf),('guard_summary.json',guard),('profile_summary.json',counters),('raw_counters.json',raw)]:
        (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    if perf:
        fields=['id','M','N','K','G','tile_n','before_ms','after_ms','delta_pct','speedup','blas_card0_ms','R']
        with (ROOT/'perf.csv').open('w') as f:
            writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(perf)
    print('VALID_JOBS',len(good))
    for tile in [128,64]:
        rows=[r for r in perf if r['tile_n']==tile]
        if rows:
            print('PERF',tile,'n',len(rows),'median_delta_pct',round(statistics.median(r['delta_pct'] for r in rows),2),
                  'slower_gt2',sum(r['delta_pct']>2 for r in rows))
            for r in rows:print('ROW',r['id'],'M',r['M'],'old',round(r['before_ms'],6),'new',round(r['after_ms'],6),'delta%',round(r['delta_pct'],1),'R',round(r['R'],3))
    for r in guard:print('GUARD',r['id'],round(r['delta_pct'],2),r['before_round_ms'],r['after_round_ms'])
    print('PROFILE_GEOMETRY_PASS',len(counters),'profiles',len(counters)*2,'dispatches')


if __name__=='__main__':main()
