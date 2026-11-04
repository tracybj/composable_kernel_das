#!/usr/bin/env python3
"""Retain every round and regression. No oracle strategy or fastest-run selection."""
import collections,csv,importlib.util,json,math,re,statistics,sys
from pathlib import Path
import run as r
P=r.ROOT
def median(v):return statistics.median(v)
def baseline():
 a={int(x['ID']):dict(ms=float(x['BLAS ms']),gpu=0,protocol='frozen Native GroupedGemm repeat20',source='/workspace/bf16NN-groupgemm.csv') for x in csv.DictReader(open('/workspace/bf16NN-groupgemm.csv',encoding='utf-8-sig'))}
 a.update({x['id']:dict(ms=x['time_ms'],gpu=4,protocol='frozen Native GroupedGemm repeat20',source='/workspace/v7_coverage/p6c/native_new15.json') for x in json.loads((P.parent/'p6c/native_new15.json').read_text())});return a
def paired(before,after,source,**kw):
 if len(before)!=3 or len(after)!=3:return
 b=[x['result']['median_ms'] for x in before];a=[x['result']['median_ms'] for x in after]
 assert all(bv.get('environment_epoch',0)==av.get('environment_epoch',0) for bv,av in zip(before,after)), 'A/B crosses environment epoch'
 row=dict(**kw,before_ms=median(b),after_ms=median(a),delta_pct=100*(median(a)/median(b)-1),before_round_ms=b,after_round_ms=a,paired_round_delta_pct=[100*(av/bv-1) for bv,av in zip(b,a)],before_cv=[x['result']['cv'] for x in before],after_cv=[x['result']['cv'] for x in after],commands=[x['key'] for pair in zip(before,after) for x in pair])
 max_cv=max(row['before_cv']+row['after_cv'])
 consistent_improvement=all(v < -2 for v in row['paired_round_delta_pct'])
 consistent_regression=all(v > 2 for v in row['paired_round_delta_pct'])
 row['unpaired_median_delta_pct']=row['delta_pct']
 row['delta_pct']=median(row['paired_round_delta_pct'])
 row['observational_label']='stable_improvement' if consistent_improvement and -row['delta_pct']>max(2,200*max_cv) else 'stable_regression' if consistent_regression and row['delta_pct']>max(2,200*max_cv) else 'uncertain'
 row['label_note']='Heuristic: all paired rounds beyond2%, median change beyond2x maximum sample CV; not a statistical confidence interval'
 row['environment_epochs']=[x.get('environment_epoch',0) for x in before]
 row['aggregation_note']='Time columns are measured medians; effect is median of within-epoch paired ratios, never a pair across recovery'
 if row.get('id') in source:
  row.update(blas=source[row['id']],blas_ms=source[row['id']]['ms'],blas_gpu=source[row['id']]['gpu'],blas_protocol=source[row['id']]['protocol'],blas_source=source[row['id']]['source'],R_before=source[row['id']]['ms']/row['before_ms'],R_after=source[row['id']]['ms']/row['after_ms'])
 return row
def write(name,rows):
 (P/(name+'.json')).write_text(json.dumps(rows,indent=2))
 if rows:
  with (P/(name+'.csv')).open('w') as f:
   fields=list(dict.fromkeys(k for x in rows for k in x if not isinstance(x[k],(list,dict))));w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def aggregate(rows):
 if not rows:return {}
 before_totals=[sum(x['before_round_ms'][n] for x in rows) for n in range(3)]
 after_totals=[sum(x['after_round_ms'][n] for x in rows) for n in range(3)]
 return dict(before_round_total_ms=before_totals,after_round_total_ms=after_totals,paired_round_total_delta_pct=[100*(av/bv-1) for bv,av in zip(before_totals,after_totals)],count=len(rows),median_delta_pct=median(x['delta_pct'] for x in rows),sum_before_ms=median(before_totals),sum_after_ms=median(after_totals),sum_delta_pct=median(100*(av/bv-1) for bv,av in zip(before_totals,after_totals)),improve_gt2=sum(x['delta_pct']< -2 for x in rows),regress_gt2=sum(x['delta_pct']>2 for x in rows),regress_all_rounds_gt2=sum(all(v>2 for v in x['paired_round_delta_pct']) for x in rows),max_regress_pct=max(x['delta_pct'] for x in rows),R_ge099_before=sum(x.get('R_before',0)>=.99 for x in rows),R_ge099_after=sum(x.get('R_after',0)>=.99 for x in rows))
def main():
 from journalview import read as read_journal
 jobs,good=read_journal();blas=baseline();cs=r.cases();out={}
 for prefix,variants,name in [('expand_cache',['base','noslc'],'cache_expanded'),('walk_cache',['noslc','noslc_row'],'cache_walk')]:
  rows=[]
  ids=sorted({x['id'] for x in good.values() if x.get('logical_key',x['key']).startswith(prefix+'_')})
  for i in ids:
   groups=[[good[f'{prefix}_r{n}_id{i}_{v}'] for n in [1,2,3] if f'{prefix}_r{n}_id{i}_{v}' in good] for v in variants]
   x=paired(*groups,blas,id=i,**{k:cs[i][k] for k in ['M','N','K','G']},policy_applies=cs[i]['K']>4096,variant=variants[1])
   if x:rows.append(x)
  write(name,rows);out[name]=aggregate([x for x in rows if x['policy_applies']]);out[name+'_diagnostic']=aggregate([x for x in rows if not x['policy_applies']])
 rows=[]
 for i in sorted({x['id'] for x in good.values() if x.get('logical_key',x['key']).startswith('smallperf_')}):
  for v in ['drain','k32','square','k32n64']:
   groups=[[good[f'smallperf_r{n}_id{i}_{v}_{tag}'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_{v}_{tag}' in good] for tag in ['before','after']]
   x=paired(*groups,blas,id=i,**{k:cs[i][k] for k in ['M','N','K','G']},variant=v,baseline_variant='old64' if v=='k32n64' else 'old128')
   if x:
    original=[good[f'smallperf_r{n}_id{i}_original']['result']['median_ms'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_original' in good]
    if len(original)==3:x.update(original_ms=median(original),after_vs_original_pct=100*(x['after_ms']/median(original)-1))
    rows.append(x)
 write('small_representatives',rows)
 for v in ['drain','k32','square','k32n64']:out['small_'+v]=aggregate([x for x in rows if x['variant']==v])
 rows=[]
 for name in sorted({x.get('contrast') for x in good.values() if x.get('logical_key',x['key']).startswith('smallexpand_')}):
  for i in sorted({x['id'] for x in good.values() if x.get('logical_key',x['key']).startswith('smallexpand_') and x.get('contrast')==name}):
   groups=[[good[f'smallexpand_{name}_r{n}_id{i}_{tag}'] for n in [1,2,3] if f'smallexpand_{name}_r{n}_id{i}_{tag}' in good] for tag in ['before','after']]
   x=paired(*groups,blas,id=i,**{k:cs[i][k] for k in ['M','N','K','G']},strategy=name,baseline_variant=groups[0][0]['variant'] if groups[0] else None,variant=groups[1][0]['variant'] if groups[1] else None)
   if x:
    original=[good[f'smallexpand_{name}_r{n}_id{i}_original'] for n in [1,2,3] if f'smallexpand_{name}_r{n}_id{i}_original' in good]
    if len(original)==3:
     x['original_ms']=median(z['result']['median_ms'] for z in original)
     x['original_round_ms']=[z['result']['median_ms'] for z in original]
     x['after_vs_original_pct']=100*(x['after_ms']/x['original_ms']-1)
     x['original_source']='current interleaved original5000 + baseline + variant; same benchmark/card4'
    elif x['baseline_variant']=='base':
     x['original_ms']=x['before_ms'];x['original_round_ms']=x['before_round_ms'];x['after_vs_original_pct']=x['delta_pct'];x['original_source']='current adjacent original5000/variant A/B; same benchmark/card4'
    rows.append(x)
  out['small_expand_'+name]=aggregate([x for x in rows if x['strategy']==name])
 write('small_expanded',rows)
 # New15 includes measured prefill switch-on controls, never invented variant times.
 newrows=[]
 for i in range(1179,1194):
  original=[good[f'smallperf_r{n}_id{i}_original'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_original' in good]
  if len(original)!=3:continue
  x=dict(id=i,**{k:cs[i][k] for k in ['M','N','K','G']},phase='decode' if cs[i]['M']<=256 else 'prefill',original_ms=median(z['result']['median_ms'] for z in original),blas_ms=blas[i]['ms'],blas_gpu=blas[i]['gpu'],blas_protocol=blas[i]['protocol'],blas_source=blas[i]['source'],original_round_ms=[z['result']['median_ms'] for z in original])
  for v in ['drain','k32','square','k32n64']:
   after=[good[f'smallperf_r{n}_id{i}_{v}_after'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_{v}_after' in good]
   if len(after)==3:
    x[v+'_ms']=median(z['result']['median_ms'] for z in after)
    x[v+'_delta_vs_original_pct']=100*(x[v+'_ms']/x['original_ms']-1)
    x[v+'_R_frozen']=blas[i]['ms']/x[v+'_ms']
    x[v+'_round_ms']=[z['result']['median_ms'] for z in after]
  for v,anchor in [('old128','drain'),('old64','k32n64')]:
   before=[good[f'smallperf_r{n}_id{i}_{anchor}_before'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_{anchor}_before' in good]
   if len(before)==3:x[v+'_ms']=median(z['result']['median_ms'] for z in before)
  controls=[good[f'smallperf_r{n}_id{i}_prefill'] for n in [1,2,3] if f'smallperf_r{n}_id{i}_prefill' in good]
  if len(controls)==3:
   x.update(prefill_switch_on_ms=median(z['result']['median_ms'] for z in controls),prefill_selected=[z['selected'] for z in controls],applicability='Outside small-M contract: M>256; measured K32 switch-on selector fallback to5000')
  else:x['applicability']='Within default-off small-M experiment correctness contract; no per-shape winner selected'
  newrows.append(x)
 write('new15',newrows)
 native=[]
 for i in [1178,76,633,681,358,772,1186,1187]:
  group=[good[f'nativeref_r{n}_id{i}_blas'] for n in [1,2,3] if f'nativeref_r{n}_id{i}_blas' in good]
  if len(group)!=3:continue
  x=dict(id=i,gpu=4,algorithm_id=group[0]['result']['algorithm_id'],kernel_name=group[0]['result']['kernel_name'],sources=[z['key'] for z in group])
  for repeat in [1,20]:
   rounds=[next(t for t in z['result']['timings'] if t['repeat']==repeat) for z in group]
   x[f'blas_repeat{repeat}_ms']=median(z['median_ms'] for z in rounds);x[f'blas_repeat{repeat}_rounds']=rounds
  x['environment_epochs']=[z.get('environment_epoch',0) for z in group]
  if i==1178:
   ck=[good[f'nativeref_r{n}_id{i}_noslc'] for n in [1,2,3] if f'nativeref_r{n}_id{i}_noslc' in good]
   if len(ck)==3:
    assert all(c.get('environment_epoch',0)==b.get('environment_epoch',0) for c,b in zip(ck,group)), 'Fresh CK/Native pair crosses environment epoch'
    x['noslc_ms']=median(z['result']['median_ms'] for z in ck);x['R_samecard_single_launch']=x['blas_repeat1_ms']/x['noslc_ms']
  native.append(x)
 write('native_reference',native)
 profiles=[];raw=[]
 pp=[x for x in good.values() if x.get('logical_key',x['key']).startswith('smallprofile_')]
 if pp:
  sys.path.insert(0,'/workspace/v7_coverage/p6d');spec=importlib.util.spec_from_file_location('p6d','/workspace/v7_coverage/p6d/analyze.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
  geom={'old128':(64,128,16),'drain':(64,128,16),'k32':(64,128,32),'old64':(64,64,16),'k32n64':(64,64,32),'square':(128,128,16)}
  for x in pp:
   ds=d.dispatches(x['perf_files'][0]);gi=d.dispatch_info(x['perf_files'][0]);assert len(ds)==len(gi)==2
   tile=geom[x['variant']];shape=tuple(x['shape'][k] for k in ['M','N','K','G']);metric=d.p6.metrics(ds[-1],shape,tile)
   for dispatch,g in zip(ds,gi):
    assert dispatch['SQ_INSTS_MMOP']==metric['mmop_insts_expected_padded'],(x['key'],'MMOP')
    assert dispatch['SQ_WAVES']==metric['ctas']*4 and g['ctas']==metric['ctas'],(x['key'],'grid/waves')
    assert g['lds']==(49152 if x['variant']=='k32' else 32768) and g['scratch']==0,(x['key'],'LDS/private',g)
   profiles.append(dict(id=x['id'],variant=x['variant'],tile=tile,geometry=gi,source=x['perf_files'][0],**metric));raw.append(dict(key=x['key'],source=x['perf_files'][0],dispatches=ds))
 write('small_profiles',profiles);(P/'small_raw_counters.json').write_text(json.dumps(raw,indent=2))
 out['jobs']=dict(attempts=len(jobs),success=len(good),raw_success=sum(x.get('ok',False) for x in jobs),excluded_raw_keys=json.loads((P/'invalidations.json').read_text()).get('raw_keys',[]) if (P/'invalidations.json').exists() else [],failures=[dict(key=x['key'],path=x['path'],rc=x.get('rc'),error=x.get('error')) for x in jobs if not x.get('ok')],by_kind=dict(collections.Counter(x.get('kind') for x in jobs if x.get('ok'))))
 (P/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
