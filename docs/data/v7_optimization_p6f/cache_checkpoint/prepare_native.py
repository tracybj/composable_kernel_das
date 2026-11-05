"""Reuse read-only Native replay logic, with this run's frozen shape metadata."""
import csv,json
from pathlib import Path
root=Path('/workspace/v7_coverage/p6f');old=Path('/workspace/v7_coverage/p6d')
cases=[]
for x in csv.DictReader(open('/workspace/bf16NN-groupgemm.csv',encoding='utf-8-sig')):
    if x['status']!='PASS':continue
    cases.append(dict(id=int(x['ID']),**{k:int(x[k]) for k in ['M','N','K','G']},algorithm_id=int(x['algorithm_id']),kernel_name=x['kernel_name'],frozen_ms=float(x['BLAS ms']),source='/workspace/bf16NN-groupgemm.csv'))
for x in json.loads((old/'cases.json').read_text()):
    if x['id']>=1179:cases.append(x)
(root/'cases.json').write_text(json.dumps(cases,indent=2))
(root/'native_replay.py').write_text((old/'blas_replay.py').read_text())
print('NATIVE_FROZEN_METADATA',len(cases),'no_retuning=true')
