from pathlib import Path
import json,csv,collections
e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt');o=e/'results/blas_winners_v116';frozen=json.loads((e/'experiments/blas_winners_v116/winners.json').read_text());result=[]
for case in frozen['results']:
 x=json.loads((o/('%04d_normal.json'%case['ID'])).read_text());assert len(x['checks'])==6;print('REPLAY',json.dumps(dict(ID=x['ID'],backend=x['winner'],algorithm=x['algorithm_id'],BLAS_TOPS=x['throughput']['BLAS'],CK_TOPS=x['throughput']['CK'],CK_to_target=x['throughput']['CK']/x['target_TOPS'],numeric_checks=len(x['checks']))))
for idx in [100,911,1126]:
 case=next(x for x in frozen['results'] if x['ID']==idx);rows=list(csv.DictReader((o/('profile_%04d'%idx)/'capture.csv').open()));br=[x for x in rows if case['kernel_name'] in x['KernelName']];cr=[x for x in rows if 'GroupedInt8Nt' in x['KernelName']];expected=3*(case['G'] if case['winner']=='matmul' else 1);assert len(br)==expected and len(cr)==3,(idx,len(br),len(cr),list(rows[0]))
 cc={};print('PMC_KEYS',idx,list(rows[0])[:35])
 for name,part in [('BLAS',br[-(case['G'] if case['winner']=='matmul' else 1):]),('CK',cr[-1:])]:
  keys=['SQ_INSTS_VALU','SQ_INSTS_SALU','SQ_INSTS_LDS','SQ_INSTS_VMEM_RD','SQ_INSTS_VMEM_WR','SQ_LDS_BANK_CONFLICT','SQ_WAIT_INST_LDS','SQ_WAIT_INST_VMEM'];counters={k:sum(int(float(x[k])) for x in part) for k in keys if k in part[0]}
  for pref in ['TCC_HIT[','TCC_MISS[']:counters[pref[:-1]]=sum(int(float(v)) for x in part for k,v in x.items() if k.startswith(pref))
  resources={k:sorted({x[k] for x in part}) for k in ['wgr','grd','lds','scr','arch_vgpr','accum_vgpr','sig','sgpr'] if k in part[0]};cc[name]=dict(launches_per_complete_call=len(part),resources=resources,counters=counters,kernel=part[0]['KernelName'])
 row=dict(ID=idx,all_profile_rows=len(rows),BLAS_total_launches=len(br),CK_total_launches=len(cr),complete_call=cc,scope='Aggregate last complete G-expert invocation;setup/reference kernels excluded. Counter values are reported aggregates,not HBM bytes or stall percentages;profile times not used for ranking.');result.append(row);print('PMC',json.dumps(row))
(o/'counter_summary.json').write_text(json.dumps(result,indent=2)+'\n')
