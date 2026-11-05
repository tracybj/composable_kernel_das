"""Small separate diagnostic; never part of formal one-launch result tables."""
import argparse,subprocess,time,sys
import runtime as r

ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--repair-tag',default='');args=ap.parse_args()
for suite,plan in [('launch_scope_accuracy','launch_scope_accuracy_frozen.json'),('launch_scope_native','launch_scope_native_frozen.json'),('launch_scope_perf','launch_scope_perf_frozen.json')]:
    path=r.ROOT/(suite+'_controller.log')
    if path.exists():
        assert args.resume;path=r.ROOT/(suite+'_controller_resume_'+str(time.time_ns())+'.log')
    cmd=[sys.executable,'-u',str(r.ROOT/'campaign.py'),suite,'--plan',str(r.ROOT/plan)]
    if args.repair_tag:cmd+=['--repair-tag',args.repair_tag]
    with path.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    r.append(r.ROOT/'launch_scope_steps.jsonl',dict(at=r.stamp(),suite=suite,rc=result.returncode,log=str(path)))
    print('LAUNCH_SCOPE',suite,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
