"""One explicit bounded trace-location repair; original failure stays immutable."""
from pathlib import Path
import copy,fcntl,hashlib,json,os,time
import plan,runtime as r,wave4_profiles_campaign as profile
ROOT=Path(__file__).resolve().parent
SELF=plan.sha(__file__)
def main():
 r.no_core()
 with (ROOT/'wave4_profile_repair.lock').open('a') as lock,(ROOT/'final_controller.lock').open('a') as final:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(final,fcntl.LOCK_EX|fcntl.LOCK_NB)
  def event(**kw):
   x=dict(at=r.stamp(),pid=os.getpid(),controller_sha256=SELF,**kw)
   r.write(ROOT/'wave4_profile_repair_status.json',x)
   with (ROOT/'wave4_profile_repair_events.jsonl').open('a') as out:out.write(json.dumps(x)+'\n');out.flush();os.fsync(out.fileno())
  try:
   assert plan.sha(__file__)==SELF
   assert time.time()+60<r.read(ROOT/'session.json')['gpu_deadline_epoch']
   assert r.read(ROOT/'post_wave4_build_status.json')['state']=='stopped'
   assert r.read(ROOT/'wave4_finish_status.json')['state']=='stopped'
   assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json','user_pause.json'])
   assert r.read(ROOT/'wave4_discovery_analysis.json')['completed_pairs']==27
   assert r.read(ROOT/'wave4_qualification_audit.json')['ok']
   rule=r.read(ROOT/'wave4_profile_rule_retry.json');retry=r.read(ROOT/'wave4_profile_retry_plan.json')
   assert rule['target_cu_order']==[3,7,15] and rule['minimum_raw_token_bytes']==4096
   assert retry['pairs']==r.read(ROOT/'wave4_profile_plan.json')['pairs']
   old=r.read(ROOT/'frozen_plan.json');assert old['phase']=='h3_wave4_discovery'
   assert old['benchmark_sha256']==plan.sha(plan.BENCH)==retry['binary_sha256']
   assert all(plan.sha(plan.REPO/n)==h for n,h in old['worktree_source_sha256'].items())
   m=copy.deepcopy(old);m.pop('plan_sha256',None)
   m.update(wave4_profile_plan_sha256=retry['sha256'],sparse_trace_repair_parent_manifest_sha256=old['plan_sha256'])
   for p in ROOT.glob('*.py'):m['runner_sha256'][p.name]=plan.sha(p)
   for name in ['wave4_profile_rule_retry.json','wave4_profile_retry_plan.json','drafts/wave4_own_commit/commit_validated_source.py']:
    m['runner_sha256'][name]=plan.sha(ROOT/name)
   m['plan_sha256']=plan.digest(m);r.write(ROOT/'manifests'/(m['plan_sha256']+'.json'),m);r.write(ROOT/'frozen_plan.json',m)
   profile.RULE=ROOT/'wave4_profile_rule_retry.json';profile.REGPLAN=ROOT/'wave4_profile_retry_plan.json'
   event(state='capturing',manifest_sha256=m['plan_sha256'],additional_CU_choices=[3,7,15])
   with r.locked():profile.capture()
   event(state='analyzing');profile.analyze()
   event(state='delivering');profile.deliver()
   names=['wave4_qualification.json','wave4_qualification_audit.json','wave4_discovery_results.json',
          'wave4_discovery_analysis.json','wave4_profiles.json','profile_delivery_batch14.json',
          'wave4_profile_rule_retry.json','wave4_profile_retry_plan.json']
   captures=r.read(ROOT/'wave4_profiles.json');assert len(captures)==16
   assert all(x['raw_token_bytes']>=4096 and x['benchmark_sha256']==m['benchmark_sha256'] for x in captures)
   r.write(ROOT/'wave4_phase_closed.json',dict(state='complete',at=r.stamp(),
       reason='explicit_bounded_sparse_trace_location_repair',epoch=r.read(ROOT/'environment_epoch.json')['id'],
       benchmark_sha256=m['benchmark_sha256'],evidence_sha256={n:plan.sha(ROOT/n) for n in names},
       original_failure_retained=True,raw_token_minimum_unchanged=4096,formal_measurements_rerun=False,
       original_CU_choices=[0,1,2],bounded_extra_CU_choices=[3,7,15],not_MKFD=True,Goal_complete=False))
   event(state='complete',note='All8registeredpairsvalid; originalthree sparseattemptsandfailedcontrollersretained; noadoption.')
  except BaseException as error:
   event(state='stopped',error=str(error),error_type=type(error).__name__,automatic_retry=False);raise
if __name__=='__main__':main()
