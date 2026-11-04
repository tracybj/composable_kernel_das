#!/usr/bin/env python3
"""Resume the frozen, built small-M campaign after a verified recovery."""
import argparse,datetime,json,resource,subprocess,sys,time
from pathlib import Path
import run as r
import journalview
P=r.ROOT
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
ap=argparse.ArgumentParser();ap.add_argument('--repair-epoch',required=True);args=ap.parse_args()
assert not (P/'environment_recovery_required.json').exists(),'Complete separate recovery first'
assert r.sha(r.BENCH)==json.loads((P/'small_binary.json').read_text())['benchmark_sha256'],'Frozen small binary changed'
_,valid=journalview.read();assert all('smallreg_'+t in valid for t in ['selector_test','device_args_query_test','device_args_smoke'])
deadline=datetime.datetime.fromisoformat(json.loads((P/'session.json').read_text())['deadline_utc']).timestamp()
def launch(cmd,name):
 if time.time()>deadline-900:raise RuntimeError('Deadline reserve')
 log=P/(name+'_'+args.repair_epoch+'.out');print('STAGE',cmd,flush=True)
 with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=r.REPO)
 if p.returncode:raise RuntimeError('STOP '+str(log))
 print('STAGE_PASS',str(log),flush=True)
for suite in ['accuracy','stress','perf','profile']:
 launch([sys.executable,str(P/'small_run.py'),suite,'--repair-epoch',args.repair_epoch],'small_'+suite)
launch([sys.executable,str(P/'run_native.py'),'--repair-epoch',args.repair_epoch],'native_reference')
launch([sys.executable,str(P/'analyze_all.py')],'analysis')
print('FORMAL_SMALL_CAMPAIGN_PASS',r.stamp(),flush=True)
