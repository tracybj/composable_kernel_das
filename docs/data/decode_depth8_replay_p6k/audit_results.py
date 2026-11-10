"""Read-only verification of actual commands/logs, gates, coverage and fresh rows."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,re,subprocess,sys
ROOT=Path(__file__).resolve().parent
TOOLS=Path('/workspace/composable_kernel/tools/grouped_gemm_depth8_reproduce')
sys.path.insert(0,str(TOOLS))
import plan,report,runtime as r

def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else []
def preservation():
 def edits(patch):
  out={};name=None
  for line in patch.splitlines():
   if line.startswith('diff --git '):name=line.split(' b/',1)[1];out[name]=[]
   elif name and line[:1] in ['+','-'] and not line.startswith(('+++','---')):out[name].append(line)
  return out
 before=edits((ROOT/'initial_tracked.patch').read_text())
 after=edits(subprocess.check_output(['git','diff','--'],cwd=plan.REPO,text=True))
 corrections={
  'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args_query_test.cpp':{
   '-    constexpr std::array<CandidateExpected, 38> expected{{':'-    constexpr std::array<CandidateExpected, 37> expected{{',
   '+    constexpr std::array<CandidateExpected, 40> expected{{':'+    constexpr std::array<CandidateExpected, 39> expected{{'},
  'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp':{
   '-inline constexpr std::array<GroupedGemmCandidateIdentity, 38>':'-inline constexpr std::array<GroupedGemmCandidateIdentity, 37>',
   '+inline constexpr std::array<GroupedGemmCandidateIdentity, 40>':'+inline constexpr std::array<GroupedGemmCandidateIdentity, 39>'}}
 for name,mapping in corrections.items():
  for source,target in mapping.items():
   assert after[name].count(source)==1,(name,source)
  after[name]=[mapping.get(line,line) for line in after[name]]
 assert before==after,'Pre-existing tracked edits changed beyond the one new registry entry'
 initial=r.read(ROOT/'initial_files_sha256.json')
 unmodified={name:digest for name,digest in initial.items() if name not in before or name in [
  'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp',
  'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp']}
 for name,digest in unmodified.items():assert plan.sha(plan.REPO/name)==digest,name
 result=dict(ok=True,tracked_files=list(before),tracked_added_deleted_sequences_preserved=True,
  unchanged_files_sha256=unmodified,registry_count_normalizations=corrections,
  note='The new instance adds one entry: old 37->39 paused edits become 38->40. Only these exact count lines are normalized; all other pre-existing added/deleted lines and unchanged files match.')
 r.write(ROOT/'preservation_audit.json',result);return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
 m=r.read(ROOT/'frozen_plan.json');assert m['plan_sha256']==plan.digest({k:v for k,v in m.items() if k!='plan_sha256'})
 for p,digest in [(plan.BENCH,m['benchmark_sha256']),(plan.BLAS,m['blas_sha256']),(plan.SELECTOR,m['selector_sha256'])]:assert plan.sha(p)==digest,str(p)
 for name,digest in m['runner_sha256'].items():assert plan.sha(TOOLS/name)==digest,name
 for name,digest in m['worktree_source_sha256'].items():assert plan.sha(plan.REPO/name)==digest,name
 for name,digest in m.get('dependencies',{}).get('sha256',{}).items():assert plan.sha(name)==digest,name
 assert plan.sha(m['table']['csv_path'])==m['table']['csv_sha256']
 for p in plan.SHAPES:assert plan.sha(p)==plan.SHAPE_SHA
 jobs=report.records(ROOT);lookup={x['key']:x for x in jobs};assert len(lookup)==len(jobs)
 session=r.read(ROOT/'session.json')
 deadline=datetime.datetime.fromtimestamp(session['deadline_epoch'],datetime.timezone.utc)
 gpu_deadline=datetime.datetime.fromtimestamp(session['gpu_deadline_epoch'],datetime.timezone.utc)
 assert all(datetime.datetime.fromisoformat(x['at'])<gpu_deadline for x in jobs)
 assert all(datetime.datetime.fromisoformat(x['finished_at'])<deadline for x in jobs)
 assert datetime.datetime.now(datetime.timezone.utc)<deadline
 starts=lines(ROOT/'started.jsonl');started={x['record']['key'] for x in starts}
 assert started==set(lookup),('UNFINISHED_PHYSICAL_JOBS',started-set(lookup))
 gates=lines(ROOT/'gates.jsonl');checks=lines(ROOT/'kernel_checks.jsonl')
 done,partial=report.completed(ROOT,m);summary=report.emit(ROOT)
 cases={x['ID']:x for x in m['table']['cases']}
 qualification=lines(ROOT/'qualification_results.jsonl')
 accepted_keys=set(k for row in done.values() for k in row['source_keys'])
 accepted_keys.update(k for row in qualification for k in row['keys'])
 regressions={x['regression_kind']:x for x in jobs if x.get('kind')=='regression' and x['ok']}
 expected_regressions={'selector_test','decode_depth8_selector_test','device_args_query_test','device_args_smoke'}
 assert set(regressions)==expected_regressions
 accepted_keys.update(x['key'] for x in regressions.values())
 comparisons=elements=0;max_abs=0.0
 for key in accepted_keys:
  x=lookup[key];assert x['ok'] and x['rc']==0 and x['physical_gpu']==4 and not x['kernel_service_events']
  assert x['benchmark_sha256']==m['benchmark_sha256'] and x['plan_sha256']==m['plan_sha256']
  assert x['environment']==dict(x['env'],HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4'),key
  gate=[g for g in gates if g['key']==key]
  assert len(gate)==1
  g=gate[0];assert g['rc']==0 and g['hcu']=='0.0%' and g['physical_gpu']==4 and g['mem_available_kib']>=64*1024*1024 and 'MKFD' not in g['output'].upper()
  for suffix in ['_pre','_post']:
   c=[z for z in checks if z['key']==key+suffix];assert len(c)==1 and c[0]['rc']==0 and not c[0]['events'],(key,suffix)
  assert x['gate_at']<=g['at']<=x['at']<=x['finished_at']
  raw=(Path(x['path'])/'run.log').read_text()
  parsed=r.parse_output(raw,x['kind'],x.get('expected'),x.get('repeats',1),x.get('regression_kind'))
  assert parsed.get('selected')==x.get('selected')
  if x['kind'] in ['accuracy','perf']:
   assert x['command'][0]==str(plan.BENCH) and '--launches-per-sample=1' in x['command'] and '--include-args-build=0' in x['command']
   assert not any(a.startswith('--instance') for a in x['command'])
   assert x['binary_sha256']==m['benchmark_sha256']
   assert {k:parsed['config'][a] for k,a in [('M','m'),('N','n'),('K','k'),('G','groups')]}=={k:str(v) for k,v in x['shape'].items()}
  if x['kind']=='accuracy':
   count=math.prod(x['shape'][k] for k in ['M','N','G'])
   assert [list(c) for c in parsed['comparisons']]==x['comparisons']
   for repeat,checked,failed,absolute in parsed['comparisons']:
    assert int(checked)==count and int(failed)==0 and math.isfinite(float(absolute))
    comparisons+=1;elements+=count;max_abs=max(max_abs,float(absolute))
  if x['kind']=='perf':assert parsed['result']==x['result']
 for i,row in done.items():
  records=[lookup[k] for k in row['source_keys']];assert len(records)==4
  accuracy=records[0];perf=records[1:]
  assert accuracy['kind']=='accuracy' and '--backend=both' in accuracy['command'] and '--output-redzone=1' in accuracy['command']
  assert accuracy['repeats']==cases[i]['accuracy_repeats'] and accuracy['repeats']==(3 if row['candidate_overlay_applied'] else 1)
  assert all(x['kind']=='perf' and x['round']==n+1 and '--backend=ck' in x['command'] and '--output-redzone=0' in x['command'] for n,x in enumerate(perf))
  assert accuracy['finished_at']<=perf[0]['at'] and all(perf[n]['finished_at']<=perf[n+1]['at'] for n in [0,1])
  assert len({x['epoch'] for x in records})==1 and len({x['attempt'] for x in records})==1
  assert row['actual_instance']==cases[i]['expected']
  assert row['env']==cases[i]['env'] and row['frozen_BLAS_ms']==cases[i]['frozen_BLAS_ms']
  assert math.isclose(row['R_frozen'],row['frozen_BLAS_ms']/row['CK_ms'],rel_tol=1e-12)
  if row['candidate_overlay_applied']:assert row['actual_instance']==6003 and plan.depth8_scope(row)
  else:assert row['actual_instance']==row['historical_expected']
 assert len(qualification)==48 and len({(x['kind'],x['ID']) for x in qualification})==48
 for row in qualification:
  qs=[lookup[k] for k in row['keys']];assert all(x['purpose']=='qualification' for x in qs)
  assert len({x['epoch'] for x in qs})==1 and len({x['attempt'] for x in qs})==1
  assert all(x['plan_sha256']==m['plan_sha256'] for x in qs)
  if row['kind']=='paired':
   assert len(qs)==8
   acc=[x for x in qs if x['kind']=='accuracy'];perf=[x for x in qs if x['kind']=='perf']
   assert len(acc)==2 and len(perf)==6 and max(x['finished_at'] for x in acc)<min(x['at'] for x in perf)
   for variant in ['before','after']:assert len([x for x in perf if x['variant']==variant])==3
 if not args.allow_partial:
  assert summary['complete'] and len(done)==1193 and not summary['incomplete_ids']
  assert summary['new_bins']['prefill']['total']==312 and summary['new_bins']['decode']['total']==881
  assert summary['overlay_measured']==6
  assert not (ROOT/'gpu_stop.json').exists() and not (ROOT/'environment_recovery_required.json').exists()
 for group in ['prefill','decode','all']:
  assert sum(summary['new_bins'][group]['bins'])==summary['new_bins'][group]['total']
 protected=subprocess.check_output(['git','diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'],cwd=plan.REPO)
 assert hashlib.md5(protected).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
 assert r.read('/workspace/v7_coverage/p6i_b_vector/stage_pause.json')['reason']=='USER_PAUSED'
 with r.locked():pass
 preservation_result=preservation()
 # Frozen runner's historical regression list uses >2%; additionally retain every positive slowdown.
 positive_regressions=[done[i] for i in sorted(done) if done[i]['historical_delta_pct']>0]
 assert {x['ID'] for x in r.read(ROOT/'all_regressions.json')}=={i for i,row in done.items() if row['historical_delta_pct']>2}
 r.write(ROOT/'all_positive_regressions.json',positive_regressions)
 result=dict(at=r.stamp(),ok=True,complete=summary['complete'],accepted_shapes=len(done),raw_jobs=len(jobs),
  accepted_jobs=len(accepted_keys),full_output_comparisons=comparisons,full_output_elements=elements,
  observed_max_abs=max_abs,physical_gpu=4,all_started_jobs_terminal=True,qualification_groups=len(qualification),
  new_bins=summary['new_bins'],high_cv_shapes=summary['high_cv_shapes'],
  failed_raw_keys=summary['invalid_keys'],partial_attempts=partial,
  p6i='paused',p6g='paused',preexisting_changes_preserved=preservation_result['ok'],
  all_positive_regressions=len(positive_regressions),regressions_gt2pct=sum(x['historical_delta_pct']>2 for x in done.values()),
  gpu_submission_deadline=gpu_deadline.isoformat(),last_gpu_submission=max(x['at'] for x in jobs),
  hard_deadline=deadline.isoformat(),deadline_checks_passed=True,protected_diff_md5=hashlib.md5(protected).hexdigest())
 r.write(ROOT/'final_audit.json',result)
 print('P6K_DATA_AUDIT_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
