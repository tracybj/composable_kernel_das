"""Finite new-instance acceptance; no implicit recovery or accuracy retries."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import runtime as r

ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--repair-tag',default='');args=ap.parse_args()
suites=[('impl_regression',None),('impl_tail_accuracy','tail_accuracy_both_tiles_frozen.json'),
        ('impl_decode_accuracy','decode_accuracy_with_baselines_frozen.json'),
        ('impl_small_cache_accuracy','small_cache_accuracy_frozen.json'),
        ('impl_native','impl_native_frozen.json'),('impl_tail_perf','tail_perf_frozen.json'),
        ('impl_decode_perf','decode_representative_perf_frozen.json'),
        ('impl_small_cache_perf','small_cache_perf_frozen.json'),
        ('impl_cache_policy','cache_policy_verification_frozen.json'),
        ('impl_tail_profiles','tail_profiles_verified_frozen.json'),
        ('impl_decode_profiles','decode_profiles_verified_frozen.json')]
# Regression generation uses the same finite official three targets.
reg=r.ROOT/'impl_regression_frozen.json'
if not reg.exists():
    import campaign
    reg.write_text(json.dumps(dict(at=r.stamp(),jobs=campaign.cache_plan('regression')),indent=2))
suites[0]=(suites[0][0],reg.name)
for suite,plan in suites:
    assert r.remaining(new=True)>300,'New-experiment cutoff reserve'
    path=r.ROOT/(suite+'_controller.log')
    if path.exists():
        assert args.resume
        path=r.ROOT/(suite+'_controller_resume_'+str(time.time_ns())+'.log')
    cmd=[sys.executable,'-u',str(r.ROOT/'campaign.py'),suite,'--plan',str(r.ROOT/plan)]
    if args.repair_tag:cmd+=['--repair-tag',args.repair_tag]
    with path.open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    r.append(r.ROOT/'phase_impl_steps.jsonl',dict(at=r.stamp(),suite=suite,rc=result.returncode,log=str(path)))
    print('IMPL_PHASE',suite,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
