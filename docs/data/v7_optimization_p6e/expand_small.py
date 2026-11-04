#!/usr/bin/env python3
"""Serial validation of predeclared small-M scopes on the frozen binary."""
import argparse,json,resource,subprocess,sys
from pathlib import Path
import run as r

resource.setrlimit(resource.RLIMIT_CORE,(0,0))
parser=argparse.ArgumentParser()
parser.add_argument('--repair-epoch',default='rec3')
args=parser.parse_args()
p=r.ROOT
assert not (p/'environment_recovery_required.json').exists()
assert r.sha(r.BENCH)==json.loads((p/'small_binary.json').read_text())['benchmark_sha256']
for name in ['k32_lowcta','drain_shortk','square_m128_256']:
    plan=p/('small_expand_plan_'+name+'.json')
    cmd=[sys.executable,str(p/'small_run.py'),'expand','--plan',str(plan),'--repair-epoch',args.repair_epoch]
    log=p/('expand_'+name+'_'+args.repair_epoch+'.out')
    print('START_SCOPE',name,flush=True)
    with log.open('w') as output:
        result=subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,cwd=r.REPO)
    if result.returncode:
        raise RuntimeError('STOP: inspect '+str(log))
    print('SCOPE_PASS',name,flush=True)
result=subprocess.run([sys.executable,str(p/'analyze_all.py')],cwd=r.REPO)
if result.returncode:
    raise RuntimeError('Expanded analysis failed')
print('EXPANDED_SMALL_CAMPAIGN_PASS',r.stamp(),flush=True)
