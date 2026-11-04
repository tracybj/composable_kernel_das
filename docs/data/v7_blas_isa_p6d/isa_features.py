#!/usr/bin/env python3
"""Derived ISA facts only; raw vendor disassembly stays outside the repository."""
import collections
import csv
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent

def main():
 extracted=json.loads((ROOT/'isa_summary.json').read_text());out=[]
 for r in extracted:
  lines=Path(r['assembly']).read_text().splitlines()
  labels=[(i,re.search('<(.*?)>:',s)[1]) for i,s in enumerate(lines) if re.search('<(.*?)>:',s)]
  row={k:r[k] for k in ['tag','symbol','code_object','address','bytes','sha256','metadata','static_instructions','direct_lds_loads','unknown_lines']}
  if 'ids' in r:row['ids']=r['ids']
  tail=next(((i,n) for i,n in labels if 'oddexit_31' in n),None)
  if tail:
   start,label=tail;end=next((i for i,n in labels if i>start),len(lines))
   ops=collections.Counter(s.strip().split()[0] for s in lines[start+1:end] if s.strip() and '//' in s)
   row['first_no_load_tail_block']=dict(label=label,line_start=start+1,line_end=end,mmac_instructions=sum(v for k,v in ops.items() if k.startswith('v_mmac')),buffer_load_instructions=sum(v for k,v in ops.items() if k.startswith('buffer_load')))
   assert row['first_no_load_tail_block']['buffer_load_instructions']==0
  # Full static kernel size includes alternative epilogues, K tails, activations,
  # and alignment padding. It is not an estimate of executed instructions.
  row['static_wait_operands']=r['wait_operands']
  loads=[s for s in lines if 'buffer_load' in s and re.search(r'\blds\b',s.split('//')[0])]
  row['direct_lds_load_modifier_counts']=dict(slc=sum(bool(re.search(r'\bslc\b',s.split('//')[0])) for s in loads),glc=sum(bool(re.search(r'\bglc\b',s.split('//')[0])) for s in loads),total=len(loads))
  out.append(row)
 (ROOT/'isa_features.json').write_text(json.dumps(out,indent=2))
 rows=list(csv.DictReader(open('/workspace/v7_coverage/p6c/merged1193.csv')))
 old=list(csv.DictReader(open('/workspace/bf16NN-groupgemm.csv',encoding='utf-8-sig')))
 new=json.load(open('/workspace/v7_coverage/p6c/native_new15.json'))
 kernels={int(r['ID']):r['kernel_name'] for r in old}|{r['id']:r['kernel_name'] for r in new}
 selected={r['symbol'] for r in out if r['tag'].startswith('blas')}
 cohorts=collections.defaultdict(list)
 for r in rows:
  m,n,k,g=(int(r[c]) for c in ['M','N','K','G'])
  name='N_tail_1232' if n==1232 else 'prefill' if m>256 else 'decode_M128_256' if m>=128 else 'decode_Mle64_Klt2048' if k<2048 else 'decode_Mle64_K2048_4096' if k<=4096 else 'decode_Mle64_Kgt4096'
  cohorts[name].append(r)
 stats=[]
 for name,rs in cohorts.items():
  stats.append(dict(cohort=name,count=len(rs),below_099=sum(float(r['R_after'])<.99 for r in rs),below_08=sum(float(r['R_after'])<.8 for r in rs),below_05=sum(float(r['R_after'])<.5 for r in rs),historical_positive_gap_ms=sum(max(float(r['CK_after_ms'])-float(r['BLAS_ms']),0) for r in rs),ids=[int(r['ID']) for r in rs]))
 result=dict(source='P6c incremental/spliced 1193, not a new full run; old BLAS GPU0, new BLAS GPU4; CK GPU4 single-launch, BLAS 20 launches/sample',selected_blas_families=len(selected),all_blas_families=len(set(kernels.values())),covered_shapes=sum(k in selected for k in kernels.values()),total_shapes=len(kernels),cohorts=stats)
 assert sum(x['count'] for x in stats)==1193
 (ROOT/'coverage.json').write_text(json.dumps(result,indent=2))
 print('ISA_FACTS_PASS',len(out),'kernels; vendor_tail_blocks',sum('first_no_load_tail_block' in r for r in out))
 print('ISA coverage',result['covered_shapes'],'/',result['total_shapes'])
 for r in stats:print({k:v for k,v in r.items() if k!='ids'})

if __name__=='__main__':main()
