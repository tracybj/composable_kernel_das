#!/usr/bin/env python3
"""Single-round preview, clearly separate from final three-round summaries."""
import argparse
import csv
import json
from pathlib import Path
from analyze import ROOT, bins, load_old

ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,required=True);args=ap.parse_args()
plan=json.loads((ROOT/'plan.json').read_text())
jobs=[json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
good={r['key']:r for r in jobs if r['ok']}
ids=plan['old_optimized_ids']+plan['new_ids'];apply=plan['old_optimized_ids']+plan['optimized_new_ids']
keys=[f'perf_r{args.round}_id{i}_n0' for i in ids]+[f'perf_r{args.round}_id{i}_n128' for i in apply]
if not all(k in good for k in keys):
 print('PREVIEW_PENDING',sum(k not in good for k in keys));raise SystemExit()
shapes={int(r['ID']):r for r in csv.DictReader(Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv').open(encoding='utf-8-sig'))}
blas={int(r['ID']):float(r['BLAS ms']) for r in csv.DictReader(Path('/workspace/bf16NN-groupgemm.csv').open(encoding='utf-8-sig'))}
for i in plan['new_ids']:blas[i]=good[f'newblas_id{i}']['result']['time_ms']
before=load_old('/workspace/v7_coverage/full1178_p4b/perf/summary.txt');after=dict(before)
for i in ids:before[i]=good[f'perf_r{args.round}_id{i}_n0']['result']['median_ms'];after[i]=before[i]
for i in apply:after[i]=good[f'perf_r{args.round}_id{i}_n128']['result']['median_ms']
rows=[]
for total in [1178,1193]:
 for label,times in [('control',before),('optimized',after)]:
  for phase in ['prefill','decode']:
   ratios=[blas[i]/ms for i,ms in times.items() if i<=total and shapes[i]['phase']==phase]
   row=dict(round=args.round,total_set=total,branch=label,phase=phase,n=len(ratios),buckets=bins(ratios),R_lt08=sum(r<.8 for r in ratios),R_lt05=sum(r<.5 for r in ratios))
   rows.append(row);print(row)
(ROOT/f'round{args.round}_preview.json').write_text(json.dumps(rows,indent=2))
