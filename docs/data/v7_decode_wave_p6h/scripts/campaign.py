"""Finite, frozen per-shape A/B; no full1193 test and no forced-instance dispatch."""
import argparse,csv,json,statistics,time,uuid,math
from pathlib import Path
import runtime as r
BENCH=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
TABLE=r.REPO/'docs/data/v7_optimization_p6f/final_results/historical_envelope_incremental_1193.csv'
ENABLE='CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_WAVE'

def shapes():
 return {int(x['ID']):dict(id=int(x['ID']),**{k:int(x[k]) for k in ['M','N','K','G']},baseline=json.loads(x['enable_env']),expected=int(x['instance_id']),historical=x) for x in csv.DictReader(TABLE.open())}
def active(c):return c['M']<=64 and c['N']%128==0 and 32<=c['K']<=4096 and c['K']%32==0 and c['G']<=1024

def plan(suite):
 cs=shapes()
 if suite=='sanity':
  return [dict(id=-i-1,M=m,N=n,K=k,G=g,baseline={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'1'},expected=5000) for i,(m,n,k,g) in enumerate([(1,128,32,1),(15,128,64,1),(16,256,96,4),(17,128,128,4),(32,256,512,4),(64,128,2048,1)])]
 if suite=='boundary_stress':
  return [dict(id=-101-i,M=m,N=n,K=k,G=g,baseline={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'1'},expected=5000) for i,(m,n,k,g) in enumerate([(1,128,32,1024),(64,128,4096,1),(17,65536,32,1)])]
 if suite=='row_geometry_stress':
  return [dict(id=-201-i,M=m,N=n,K=k,G=g,baseline={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'1'},expected=5000) for i,(m,n,k,g) in enumerate([(31,128,64,1),(33,256,128,4),(49,128,512,64)])]
 if suite=='configuration_min_stage':
  return [dict(id=-301-i,M=m,N=128,K=k,G=4,baseline={'CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN':'1'},expected=5000) for i,(m,k) in enumerate([(1,32),(1,128),(17,32),(17,128),(33,32),(33,64),(49,96)])]
 if suite=='representatives':return [cs[i] for i in [76,633,681,620,621,358,772,*range(1179,1194)]]
 if suite=='low_group_short_small':return [c for c in cs.values() if c['M']<=16 and c['N']<=4096 and 512<=c['K']<=2048 and c['G']<=16 and active(c)]
 raise ValueError(suite)

def bench(c,variant,kind,key,repeat=3):
 env=dict(c['baseline']);expected=c['expected']
 if variant=='new':
  env[ENABLE]='1'
  if active(c):expected=6000
 timing=kind=='perf'
 cmd=[str(BENCH),'--dtype=bf16','--layout=NN',*[f'--{arg}={c[k]}' for arg,k in [('m','M'),('n','N'),('k','K'),('groups','G')]],'--backend='+('ck' if timing else 'both'),'--blas-lib='+r.BLAS,'--include-args-build=0','--validate=1','--warmup=10','--samples=30','--launches-per-sample=1',f'--time={int(timing)}',f'--correctness-repeats={1 if timing else repeat}',f'--output-redzone={0 if timing else 1}']
 return dict(key=key,id=c['id'],shape={k:c[k] for k in ['M','N','K','G']},variant=variant,kind=kind,expected=expected,command=cmd,env=env)

def run(suite):
 cases=plan(suite);binary=r.sha(BENCH)
 frozen=r.ROOT/('plan_'+suite+'_'+binary[:12]+'.json')
 spec=dict(suite=suite,table_sha256=r.sha(TABLE),binary_sha256=binary,cases=cases)
 if frozen.exists():assert json.loads(frozen.read_text())==spec,'Frozen plan/binary changed: use a new suite/version'
 else:frozen.write_text(json.dumps(spec,indent=2))
 done=[]
 p=r.ROOT/'shape_results.jsonl'
 if p.exists():done=[json.loads(x) for x in p.read_text().splitlines()]
 with r.locked():
  for c in cases:
   if any(x['suite']==suite and x['id']==c['id'] and x['binary_sha256']==binary for x in done):continue
   attempt=uuid.uuid4().hex[:12];records=[];epoch=r.epoch()
   accuracy_only=suite in ['sanity','boundary_stress','row_geometry_stress','configuration_min_stage']
   for variant in (['new'] if accuracy_only else ['base','new']):
    job=bench(c,variant,'accuracy',f'{suite}_{c["id"]}_{attempt}_{variant}_acc',repeat=20 if accuracy_only and suite!='sanity' else 3)
    records.append(r.execute(job));assert r.sha(BENCH)==binary
   if accuracy_only:
    row=dict(suite=suite,id=c['id'],attempt=attempt,epoch=epoch,binary_sha256=binary,accuracy_only=True,keys=[x['key'] for x in records])
   else:
    times={'base':[],'new':[]};cvs={'base':[],'new':[]}
    for round_number in range(1,4):
     for variant in ['base','new']:
      rec=r.execute(bench(c,variant,'perf',f'{suite}_{c["id"]}_{attempt}_{variant}_r{round_number}'));records.append(rec)
      times[variant].append(rec['result']['median_ms']);cvs[variant].append(rec['result']['cv'])
      assert r.sha(BENCH)==binary
    assert all(x['epoch']==epoch for x in records),'No cross-epoch accepted group'
    base=statistics.median(times['base']);new=statistics.median(times['new'])
    row=dict(suite=suite,id=c['id'],shape={k:c[k] for k in ['M','N','K','G']},active=active(c),attempt=attempt,epoch=epoch,binary_sha256=binary,times=times,cvs=cvs,base_ms=base,new_ms=new,change_pct=(new/base-1)*100,keys=[x['key'] for x in records],frozen_BLAS_ms=c['historical']['frozen_BLAS_ms'],BLAS_source=c['historical']['BLAS_source'],R_new=float(c['historical']['frozen_BLAS_ms'])/new)
   r.append(p,row);print('SHAPE_COMPLETE',json.dumps(row),flush=True)
 print('SUITE_PASS',suite,len(cases),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('suite',choices=['sanity','boundary_stress','row_geometry_stress','configuration_min_stage','representatives','low_group_short_small']);run(a.parse_args().suite)
