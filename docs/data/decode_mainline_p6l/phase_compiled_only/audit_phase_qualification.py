"""Read-only audit of all registered factor qualification raw records."""
from pathlib import Path
import hashlib,json,math
import runtime as r,plan
ROOT=Path(__file__).resolve().parent

def main():
 records_all=r.read(ROOT/'phase_qualification.json');assert records_all
 manifest=r.read(ROOT/'manifests'/(records_all[0]['plan_sha256']+'.json'));assert manifest['phase']=='h3_phase_discovery'
 epoch=r.read(ROOT/'environment_epoch.json')['id']
 frozen=r.read(ROOT/'phase_discovery_plan.json')
 assert plan.sha(ROOT/'phase_discovery_plan.json')==manifest['runner_sha256']['phase_discovery_plan.json']
 assert frozen['sha256']==manifest['candidate_plan_sha256']
 copy=dict(frozen);copy.pop('sha256');assert plan.digest(copy)==frozen['sha256']
 records=r.read(ROOT/'phase_qualification.json')
 records=[x for x in records if x['plan_sha256']==manifest['plan_sha256'] and x['epoch']==epoch]
 assert len(records)==27 and len({x['logical_key'] for x in records})==27
 jobs=[json.loads(line) for line in (ROOT/'jobs.jsonl').read_text().splitlines() if line]
 bykey={x['key']:x for x in jobs};assert len(jobs)==len(bykey)
 cases={f'phase_qual_{c["ID"]}_{c["variant"]}':c for c in frozen['qualification_cases']}
 assert len(cases)==25
 gates={x['key']:x for x in [json.loads(z) for z in (ROOT/'gates.jsonl').read_text().splitlines()]}
 kernels={x['key']:x for x in [json.loads(z) for z in (ROOT/'kernel_checks.jsonl').read_text().splitlines()]}
 checks=[];comparison_count=0;elements=0;max_abs=0.0
 for record in records:
  assert bykey[record['key']]==record
  gate=gates[record['key']];assert gate['rc']==0 and gate['physical_gpu']==4 and gate['hcu']=='0.0%' and gate['mem_available_kib']>=64*1024*1024 and 'MKFD' not in gate['output'].upper()
  assert all(kernels[record['key']+suffix]['rc']==0 and not kernels[record['key']+suffix]['events'] for suffix in ['_pre','_post'])
  assert record['ok'] and record['physical_gpu']==4 and record['epoch']==epoch
  assert record['benchmark_sha256']==manifest['benchmark_sha256'] and not record['kernel_service_events']
  assert record['environment']['HIP_VISIBLE_DEVICES']==record['environment']['CUDA_VISIBLE_DEVICES']=='4'
  text=(Path(record['path'])/'run.log').read_text()
  if record['kind']=='regression':
   binary=Path(record['binary']);assert plan.sha(binary)==record['binary_sha256']
   kind='device_args_query_test' if binary.name.endswith('query_test') else 'device_args_smoke'
   r.parse_output(text,'regression',regression=kind)
   assert not [k for k in record['environment'] if k.startswith(plan.PREFIX)]
   continue
  assert record['kind']=='accuracy' and record['logical_key'] in cases
  case=cases[record['logical_key']];expected=case['expected']
  assert record['ID']==case['ID'] and record['shape']=={z:case[z] for z in ['M','N','K','G']}
  assert record['expected']==record['selected']==expected and record['repeats']==3
  assert record['binary_sha256']==manifest['benchmark_sha256']
  assert {k:v for k,v in record['environment'].items() if k.startswith(plan.PREFIX)}==case['env']
  parsed=r.parse_output(text,'accuracy',expected,3)
  assert [list(x) for x in parsed['comparisons']]==record['comparisons']
  count=case['M']*case['N']*case['G']
  for repeat,actual,failed,error in parsed['comparisons']:
   assert int(actual)==count and int(failed)==0 and math.isfinite(float(error))
   comparison_count+=1;elements+=int(actual);max_abs=max(max_abs,float(error))
  checks.append(dict(ID=case['ID'],variant=case['variant'],expected_instance=expected,shape=record['shape'],job_key=record['key'],full_comparisons=3,elements_per_comparison=count,raw_log_sha256=hashlib.sha256((Path(record['path'])/'run.log').read_bytes()).hexdigest()))
 assert len(checks)==25 and comparison_count==75
 report=dict(at=r.stamp(),ok=True,qualification_jobs=27,accuracy_jobs=25,full_output_comparisons=comparison_count,full_output_elements=elements,failed_elements=0,max_abs=max_abs,physical_gpu=4,epoch=epoch,binary_sha256=manifest['benchmark_sha256'],plan_sha256=manifest['plan_sha256'],qualification_sha256=plan.sha(ROOT/'phase_qualification.json'),no_inherited_experiment_flags=True,real_guard_and_redzone=True,rows=checks,not_performance_acceptance=True)
 r.write(ROOT/'phase_qualification_audit.json',report)
 print('H3_FACTORS_FULL_QUALIFICATION_RAW_AUDIT_PASS',27,comparison_count,elements,'failed0','max_abs',max_abs,flush=True)

if __name__=='__main__':main()
