"""One local command for freeze/resume, seven-block measurement, and report.

Preparation only until explicitly invoked with a real evidence-backed proposal.
Never restarts the ongoing discovery queue or silently draws another seed.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time,uuid
ROOT=Path(__file__).resolve().parent
SELF=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposal',required=True);ap.add_argument('--new-property-attempt',action='store_true');args=ap.parse_args()
 # The outer controller lock is separate from the GPU lock acquired by children.
 # Keep this descriptor alive across freeze/run/report to prevent double-clicks
 # from creating two frozen attempts between their individual GPU lock scopes.
 controller_lock=(ROOT/'final_controller.lock').open('a')
 fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 proposal_path=Path(args.proposal).resolve();proposal=read(proposal_path);proposal_sha=sha(proposal_path)
 assert proposal_path.is_relative_to(ROOT),'Keep a reviewable local proposal in this session'
 assert read(ROOT/'compile/build9/completed.json')['benchmark_sha256']==sha('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
 assert read(ROOT/'phase_build_isa_audit.json')['ok']
 assert read(ROOT/'phase_controller_status.json')['state']=='complete'
 assert len(read(ROOT/'phase_discovery_results.json'))==46
 assert read(ROOT/'phase_qualification_audit.json')['ok']
 assert (ROOT/'profile_delivery_batch9.json').exists(),'Deliver registered resource perf first'
 assert read(ROOT/'attempt1_closure.json')['whole_family_rejected']
 folder=ROOT/'final_controllers'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir(parents=True,exist_ok=False)
 def event(**kw):
  row=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),controller_sha256=SELF,proposal_sha256=proposal_sha,**kw)
  with (folder/'events.jsonl').open('a') as out:out.write(json.dumps(row)+'\n');out.flush();os.fsync(out.fileno())
  temporary=folder/'status.tmp';temporary.write_text(json.dumps(row,indent=2)+'\n');temporary.replace(folder/'status.json')
 def guard(gpu):
  assert sha(__file__)==SELF and sha(proposal_path)==proposal_sha
  session=read(ROOT/'session.json');assert time.time()<session['deadline_epoch'],'Original hard deadline reached'
  if gpu:
   assert time.time()+60<session['gpu_deadline_epoch'],'Original new-GPU cutoff reached'
   assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json','user_pause.json'])
 def command(stage,gpu=False):
  guard(gpu);cmd=[sys.executable,'-B','-u',str(ROOT/'final_confirmation.py'),stage]
  if stage=='freeze':cmd+=['--proposal',str(proposal_path)]
  event(state='running',stage=stage,command=cmd);start=time.monotonic()
  child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL);rc=child.wait()
  event(state='passed' if rc==0 else 'failed',stage=stage,rc=rc,child_pid=child.pid,seconds=time.monotonic()-start)
  return rc
 guard(True);active=ROOT/'final_active.json';manifest=read(ROOT/'frozen_plan.json');epoch=read(ROOT/'environment_epoch.json')['id']
 if active.exists():
  prior=read(Path(read(active)['path'])/'plan.json')
  assert prior['epoch']==epoch,'Epoch changes require separate audited recovery, not an automatic retry'
  if prior['benchmark_sha256']!=manifest['benchmark_sha256']:
   promotion=read(ROOT/'resource_promotion.json')
   prior_build=read(ROOT/'compile/build9/reference_shaders.json')
   current_build=read(ROOT/'compile/build9/completed.json')
   assert prior_build['binary_sha256']==promotion['new_benchmark_sha256']
   assert args.new_property_attempt and prior['number']==1 and manifest['phase']=='h3_phase_discovery'
   assert prior['benchmark_sha256']==promotion['old_benchmark_sha256'] and manifest['benchmark_sha256']==current_build['benchmark_sha256']
   assert read(ROOT/'attempt1_closure.json')['whole_family_rejected']
   assert prior['proposal_sha256']!=proposal_sha,'Never reuse unchanged candidate with a new binary to resample'

  if prior['proposal_sha256']==proposal_sha:
   assert not args.new_property_attempt,'An unchanged candidate cannot consume another heldout seed'
   assert manifest['phase']=='final_confirmation' and manifest['final_plan_sha256']==prior['sha256']
   event(state='resuming_existing_attempt',attempt=prior['number'],plan_sha256=prior['sha256'])
  else:
   assert args.new_property_attempt,'Changed proposal requires an explicit new property attempt'
   def properties(families):
    return {name:dict(candidate_ids=sorted(set(value['candidate_ids'])),env={k:v for k,v in value['env'].items() if v!='0'}) for name,value in families.items()}
   assert properties(proposal['families'])!=properties(prior['proposal']['families']),'Do not change explanation, ID ordering, or explicit zero flags to resample an unchanged candidate'
   assert command('freeze',True)==0
 else:
  assert not args.new_property_attempt,'The first attempt uses the first frozen seed'
  assert manifest['phase']=='h3_phase_discovery'
  assert command('freeze',True)==0
 rc=command('run',True)
 # A partial run may still have auditable results. Reporting submits no GPU job.
 report_rc=command('report',False)
 if rc or report_rc:
  event(state='stopped',measurement_rc=rc,report_rc=report_rc,note='Retain all records. No automatic retry, seed reset, or Goal completion.')
  raise SystemExit(rc or report_rc)
 current=Path(read(ROOT/'final_active.json')['path']);report=read(current/'report.json')
 event(state='measurement_and_report_complete',report=str(current/'report.json'),goal_performance_pass=report['goal_performance_pass'],stable_weak_originals=report['stable_weak_originals'],attributable_crossings=report['attributable_crossings'],note='Do not mark the Goal complete without auditing every remaining task and deliverable.')
 print('FINAL_ONE_COMMAND_FINISHED',current,'performance_goal_pass',report['goal_performance_pass'],flush=True)

if __name__=='__main__':main()
