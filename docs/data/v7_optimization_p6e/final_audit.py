#!/usr/bin/env python3
"""Campaign completion assertions against recorded commands, logs and live source."""
import collections,hashlib,json,re,subprocess
from pathlib import Path
import run as r
P=r.ROOT;R=r.REPO

def main():
 from journalview import read as read_journal
 jobs,good=read_journal();gates=[json.loads(x) for x in (P/'gate.jsonl').read_text().splitlines()];evidence={}
 def require(name,condition,detail=None):
  evidence[name]=dict(passed=bool(condition),detail=detail)
  assert condition,(name,detail)
 try:
  protected='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp';diff=subprocess.check_output(['git','diff','--',protected],cwd=R)
  require('protected_readfirstlane_preserved',hashlib.md5(diff).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a')
  require('protected_not_staged',not subprocess.check_output(['git','diff','--cached','--',protected],cwd=R))
  inputs=json.loads((P/'inputs.json').read_text());pins=[(p,m) for p,m in inputs.items() if 'benchmark' not in p]
  require('frozen_inputs_unchanged',all(r.sha(p)==m['sha256'] for p,m in pins),[p for p,_ in pins])
  require('two_active_shape_files_identical',Path('/workspace/shapes_1178.csv').read_bytes()==Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv').read_bytes())
  native_manifest=json.loads((P.parent/'p6d/manifest.json').read_text())
  native_pins={path:meta for path,meta in native_manifest.items() if path.endswith('TensileLibrary_gfx936.co') or path.endswith('native_new15.json')}
  require('frozen_native_dependencies_unchanged',len(native_pins)==2 and all(r.sha(path)==meta['sha256'] for path,meta in native_pins.items()),native_pins)
  abi=json.loads((P/'abi_audit.json').read_text());require('abi_source_audit',abi['public_v1_header_unchanged_outside_instance_enum'] and abi['existing_ids_preserved'] and abi['GemmTransKernelArgImpl_defining_header_unchanged'],abi)
  isa=json.loads((P/'cache_isa.json').read_text());require('original_five_shaders_unchanged',len(isa['original_shaders_unchanged'])==5)
  require('cache_only_slc_bits',len(isa['cache_pairs'])==2 and all(x['only_slc_machine_bits_differ'] for x in isa['cache_pairs']))
  require('four_small_shaders_present',len(isa.get('small_resources',[]))==4,[x['variant'] for x in isa.get('small_resources',[])])
  require('new_shaders_no_spill',all(x['metadata'].get('.private_segment_fixed_size')==0 and x['metadata'].get('.sgpr_spill_count')==0 and x['metadata'].get('.vgpr_spill_count')==0 for x in isa['small_resources']))
  latest_gates={x['key']:x for x in gates}
  healthy=lambda x:x['hcu4']=='0.0%' and x['rc']==0 and 'MKFD' not in x['output'] and x['mem_available_kib']>=64*1024*1024
  require('accepted_jobs_have_healthy_gates',all(x['key'] in latest_gates and healthy(latest_gates[x['key']]) for x in good.values()),dict(raw_gate_count=len(gates),retries=sum(x.get('attempt',1)>1 for x in gates)))
  bad_gates=[x for x in gates if not healthy(x)]
  recoveries=[json.loads(x) for x in (P/'recovery.jsonl').read_text().splitlines()] if (P/'recovery.jsonl').exists() else []
  epoch_proofs={x['environment_epoch']:x for x in recoveries if x.get('ok')}
  require('no_unhandled_environment_transition',not (P/'environment_recovery_required.json').exists())
  if bad_gates:
   require('recorded_full_recovery',bool(epoch_proofs))
   for epoch,proof in epoch_proofs.items():
    checks=proof['checks'];gap=(__import__('datetime').datetime.fromisoformat(checks[1]['at'])-__import__('datetime').datetime.fromisoformat(checks[0]['at'])).total_seconds()
    prior=next(x for x in jobs if x['key']==proof['known_good_prior_key'] and x.get('ok'))
    log=Path(proof['path'],'run.log').read_text()
    require('recovery_epoch_'+str(epoch),len(checks)==2 and all(healthy(x) for x in checks) and gap>=60 and proof['benchmark_sha256']==prior['benchmark_sha256'] and 'SELECT instance_id=5000' in log and 'SUMMARY result=PASS' in log and len(re.findall(r'ACCURACY backend=ck_vs_blas .*?failed=0(?:\s|$)',log))==3,dict(proof=proof['key'],gap_seconds=gap))
   transitions=json.loads((P/'environment_transitions.json').read_text())['events']
   ends={x['environment_epoch']:x['began_at'] for x in transitions}
   def in_epoch(x):
    dt=__import__('datetime').datetime;td=__import__('datetime').timedelta
    epoch=x.get('environment_epoch',0);start=dt.fromisoformat(x['at']);end=start+td(seconds=x['seconds'])
    if epoch>0 and (epoch not in epoch_proofs or start<dt.fromisoformat(epoch_proofs[epoch]['at'])):return False
    return epoch not in ends or end<dt.fromisoformat(ends[epoch])
   require('no_unvalidated_post_transition_results',all(in_epoch(x) for x in good.values()))
   inv=json.loads((P/'invalidations.json').read_text());require('invalidated_raw_values_retained',all(Path(x['path'],'run.log').exists() for x in jobs if x['key'] in inv['raw_keys']),inv)
  gatekeys={x['key'] for x in gates}
  require('gate_for_every_recorded_gpu_job',all(x['key'] in gatekeys for x in jobs))
  rawpositions={x['key']:n for n,x in enumerate(jobs) if x.get('ok')}
  positions={key:rawpositions[x['key']] for key,x in good.items()}
  alternating=[]
  for k,x in good.items():
   if k.startswith('expand_cache_') and k.endswith('_base'):alternating.append((k,k[:-5]+'_noslc'))
   if k.startswith(('smallperf_','smallexpand_')) and k.endswith('_before'):alternating.append((k,k[:-7]+'_after'))
  require('adjacent_alternating_AB',bool(alternating) and all(a in positions and b in positions and positions[b]==positions[a]+1 for a,b in alternating),dict(pairs=len(alternating)))
  require('all_AB_pairs_same_environment_epoch',all(good[a].get('environment_epoch',0)==good[b].get('environment_epoch',0) for a,b in alternating))
  observation=json.loads((P/'final_environment_observation.json').read_text())
  require('final_service_log_check',observation['accepted_jobs']==len(good) and observation['all_accepted_jobs_finished_before_observation'] and not observation['accepted_job_service_interruptions'] and not observation['paired_service_interruptions'],observation)
  kinds=collections.Counter(x['kind'] for x in good.values());require('cache_accuracy_pass',sum(x.get('logical_key',x['key']).startswith('acc_cache_') for x in good.values())==13)
  require('all_expanded_cache_pairs',sum(k.startswith('expand_cache_') for k in good)==1386)
  require('small_accuracy_and_stress_present',sum(k.startswith('smallacc_') for k in good)==161 and sum(k.startswith('smallstress_') for k in good)==8)
  for v in ['drain','k32','square','k32n64']:
   accuracy=[x for x in good.values() if x.get('logical_key',x['key']).startswith(('smallacc_','smallstress_')) and x['variant']==v]
   perf=[x for x in good.values() if x.get('logical_key',x['key']).startswith(('smallperf_','smallexpand_')) and x['variant']==v]
   require('precision_precedes_perf_'+v,bool(accuracy) and bool(perf) and max(x['at'] for x in accuracy)<min(x['at'] for x in perf))
  require('small_representative_jobs',sum(k.startswith('smallperf_') for k in good)==603)
  require('small_profiles_17',sum(k.startswith('smallprofile_') for k in good)==17)
  diagnostic_plan=json.loads((P/'square_diagnostic_plan.json').read_text())
  diagnostic_jobs=[x for k,x in good.items() if k.startswith('squarediag_')]
  diagnostic_profiles=json.loads((P/'square_diagnostic_profiles.json').read_text())
  require('square_followup_four_profiles',len(diagnostic_jobs)==len(diagnostic_profiles)==4 and diagnostic_plan['planned_at']<min(x['at'] for x in diagnostic_jobs) and all(x['mmop_count_matches_padded'] for x in diagnostic_profiles))
  for i in diagnostic_plan['ids']:
   before=f'squarediag_id{i}_before';after=f'squarediag_id{i}_after'
   require('square_followup_pair_'+str(i),positions[after]==positions[before]+1 and good[before]['environment_epoch']==good[after]['environment_epoch'])
  require('regression_three',all('smallreg_'+t in good for t in ['selector_test','device_args_query_test','device_args_smoke']))
  querylog=Path(good['smallreg_device_args_query_test']['path'],'run.log').read_text()
  require('actual_C_api_byte_boundaries', 'QUERY_V7_BYTE_RANGE_PASS cases=7' in querylog)
  require('native_frozen_reference_27',sum(k.startswith('nativeref_') for k in good)==27)
  frozen_cases={x['id']:x for x in json.loads((P.parent/'p6d/cases.json').read_text())}
  require('native_cases_match_previous_commit',(P.parent/'p6d/cases.json').read_bytes()==(R/'docs/data/v7_blas_isa_p6d/cases.json').read_bytes())
  for i in [1178,76,633,681,358,772,1186,1187]:
   native_jobs=[good[f'nativeref_r{n}_id{i}_blas'] for n in [1,2,3]]
   require('native_exact_frozen_algorithm_'+str(i),all(x['result']['algorithm_id']==frozen_cases[i]['algorithm_id'] and x['result']['kernel_name']==frozen_cases[i]['kernel_name'] and all(token in x['command'] for token in ['--warmup=10','--samples=30','--repeats=1,20']) for x in native_jobs))
  decisions=json.loads((P/'experiment_decisions.json').read_text())
  require('all_directions_disposition',all(k in decisions for k in ['cache','cache_walk','drain','k32','k32n64','square']))
  predicates={
   'k32_lowcta':lambda c:c['M']<=64 and 128<=c['K']<=4096 and c['K']%32==0 and c['N']%128==0 and c['G']<=1024 and ((c['M']+63)//64)*((c['N']+127)//128)*c['G']<80,
   'drain_shortk':lambda c:c['M']<=64 and 64<=c['K']<2048 and c['K']%16==0 and c['N']%128==0 and c['G']<=1024,
   'square_m128_256':lambda c:128<=c['M']<=256 and 64<=c['K']<=4096 and c['K']%16==0 and c['N']%128==0 and c['G']<=1024}
  require('three_predeclared_small_scopes',sorted(p.stem.replace('small_expand_plan_','') for p in P.glob('small_expand_plan_*.json'))==sorted(predicates))
  for planfile in P.glob('small_expand_plan_*.json'):
   plan=json.loads(planfile.read_text());name=plan['name']
   selected=[x for x in good.values() if x.get('logical_key',x['key']).startswith('smallexpand_'+name+'_')]
   require('expanded_small_'+name,len(selected)==len(plan['ids'])*(9 if plan.get('include_original') else 6),dict(plan=str(planfile),shape_count=len(plan['ids']),jobs=len(selected)))
   require('predeclared_before_measurement_'+name,plan['planned_at']<min(x['at'] for x in selected))
   require('full_attribute_domain_'+name,plan['ids']==sorted(i for i,c in r.cases().items() if predicates[name](c)))
   require('frozen_expansion_binary_'+name,all(x['benchmark_sha256']==plan['benchmark_sha256'] for x in selected))
  expanded=json.loads((P/'cache_expanded.json').read_text());new=json.loads((P/'new15.json').read_text());small=json.loads((P/'small_representatives.json').read_text());profiles=json.loads((P/'small_profiles.json').read_text())
  require('cache_212_primary_and19_controls',len(expanded)==231 and sum(x['policy_applies'] for x in expanded)==212)
  require('all_new15_in_results',[x['id'] for x in new]==list(range(1179,1194)))
  require('new_prefill_fallback_controls',all(x.get('prefill_selected')==[5000]*3 for x in new if x['M']>256))
  require('small_84_three_round_contrasts',len(small)==84 and all(len(x['before_round_ms'])==len(x['after_round_ms'])==3 for x in small))
  require('small_profile_geometry_and_mmop',len(profiles)==17 and all(x['mmop_count_matches_padded'] for x in profiles),dict(profiles=len(profiles),dispatches=2*len(profiles)))
  smallexpanded=json.loads((P/'small_expanded.json').read_text())
  require('small_expansion_complete_with_original_and_blas',len(smallexpanded)==310 and all(all(k in x for k in ['original_ms','original_source','blas_ms','blas_source','before_round_ms','after_round_ms','environment_epochs']) for x in smallexpanded))
  permitted=set(json.loads((P/'cache_expand_plan.json').read_text())['ids'])|set(json.loads((P/'small_perf_plan.json').read_text())['ids'])
  for planfile in P.glob('small_expand_plan_*.json'):permitted.update(json.loads(planfile.read_text())['ids'])
  require('gpu_scope_matches_predeclared_plans',all(x.get('id',0)<=0 or x['id'] in permitted for x in good.values()))
  measured={x['id'] for x in good.values() if x.get('id',0)>0};require('no_1193_full_replay',len(measured)<1193,dict(unique_ids=len(measured),ids=sorted(measured)))
  allcommands=[x['command'] for x in jobs if x.get('kind') in ['accuracy','perf','profile']]
  require('no_forced_kernel_id',all(not any('force' in token.lower() and ('instance' in token.lower() or 'kernel' in token.lower()) for token in cmd) for cmd in allcommands))
  require('formal_benchmark_entry',all(str(r.BENCH) in cmd for cmd in allcommands))
  for prefix in ['expand_cache','smallperf','smallexpand']:
   records=[x for x in good.values() if x.get('logical_key',x['key']).startswith(prefix+'_')]
   require('single_binary_'+prefix,len({x['benchmark_sha256'] for x in records})==1)
  require('current_benchmark_hash_pinned',r.sha(r.BENCH)==json.loads((P/'small_binary.json').read_text())['benchmark_sha256'])
  require('no_new_tail_epilogue_edits',not subprocess.check_output(['git','diff','83c8c2a','--','example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_v7_small_m_epilogue.hpp','example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_v7_uniform_epilogue.hpp','include/ck_tile/ops/gemm/epilogue'],cwd=R))
  for v in ['drain','k32','square','k32n64']:
   rec=good['smallacc_'+v+'_n1232'];require('n1232_fallback_'+v,rec['selected']==4003)
  failed=[x for x in jobs if not x.get('ok')]
  require('failures_retained',all(Path(x['path'],'run.log').exists() for x in failed),[dict(key=x['key'],path=x['path'],rc=x.get('rc'),error=x.get('error')) for x in failed])
  evidence['summary']=dict(passed=True,attempts=len(jobs),successful=len(good),raw_successful=sum(x.get('ok',False) for x in jobs),invalidated_keys=len(json.loads((P/'invalidations.json').read_text())['raw_keys']) if (P/'invalidations.json').exists() else 0,environment_epochs=sorted({x.get('environment_epoch',0) for x in good.values()}),by_kind=dict(kinds),full_comparisons=sum(len(x.get('comparisons',[])) for x in good.values()),full_elements=sum(int(c[1]) for x in good.values() for c in x.get('comparisons',[])),max_abs_errors=sorted({c[3] for x in good.values() for c in x.get('comparisons',[])}))
  print('CAMPAIGN_AUDIT_PASS',json.dumps(evidence['summary']))
 finally:(P/'final_audit.json').write_text(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
