#!/usr/bin/env python3
"""Sequential continuation: never relink a binary during a measured campaign."""
import datetime,fcntl,json,resource,subprocess,sys,time
from pathlib import Path
import run as r
P=r.ROOT
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def launch(cmd,log):
 print('STAGE',cmd,flush=True)
 with (P/log).open('w') as f:p=subprocess.run(cmd,cwd=r.REPO,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('STOP '+str(P/log))
 print('STAGE_PASS',log,flush=True)
end=datetime.datetime.fromisoformat(json.loads((P/'session.json').read_text())['experiment_cutoff_utc']).timestamp()
print('WAIT_CACHE_AND_LIBRARY',r.stamp(),flush=True)
while True:
 c=(P/'cache_expand_recovery2.out').read_text() if (P/'cache_expand_recovery2.out').exists() else ''
 b=(P/'build_small.log').read_text() if (P/'build_small.log').exists() else ''
 if 'Traceback ' in c or 'Error ' in b or 'error:' in b:raise RuntimeError('Pending stage failed, inspect logs')
 if 'ALL_DONE cache_expand' in c and 'Built target ck_tile_hcu_grouped_gemm\n' in b:break
 if time.time()>end-1800:raise RuntimeError('Too late to start next experiments')
 time.sleep(15)
from journalview import read as read_journal
records,valid=read_journal()
cache={k:x for k,x in valid.items() if k.startswith('expand_cache_')}
assert len(cache)==1386,len(cache)
for key,x in cache.items():
 if key.endswith('_base'):
  assert x.get('environment_epoch',0)==cache[key[:-5]+'_noslc'].get('environment_epoch',0),'Cross-epoch A/B'
assert not (P/'environment_recovery_required.json').exists()
(P/'cache_acceptance_checkpoint.json').write_text(json.dumps(dict(at=r.stamp(),logical_jobs=len(cache),environment_epochs=sorted({x.get('environment_epoch',0) for x in cache.values()}),raw_invalidations=json.loads((P/'invalidations.json').read_text()),all_pairs_within_epoch=True),indent=2))
cache_hash=r.sha(P/'bench_cache')
assert cache_hash=='1052b95c681371b0713c357ade4e3ede659df08dd4d763e3cf6023e9ab5d0d2d'
assert all(x['benchmark_sha256']==cache_hash for x in cache.values())
gates=[json.loads(s) for s in (P/'gate.jsonl').read_text().splitlines()]
latest_gates={x['key']:x for x in gates}
assert all(latest_gates[x['key']]['rc']==0 and latest_gates[x['key']]['hcu4']=='0.0%' for x in valid.values()),'Accepted job lacks healthy gate'
recoveries=[json.loads(x) for x in (P/'recovery.jsonl').read_text().splitlines()]
assert any(x.get('ok') and x.get('environment_epoch')==r.current_epoch() for x in recoveries),'Recovery proof missing'
with Path('/workspace/v7_coverage/p6b/run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 launch(['cmake','--build','build','--target',*['tile_example_grouped_gemm_'+x for x in ['device_args_benchmark','selector_test','device_args_query_test','device_args_smoke']],'-j4'],'build_small_final.log')
(P/'bench_small').write_bytes(r.BENCH.read_bytes())
(P/'small_binary.json').write_text(json.dumps(dict(at=r.stamp(),benchmark_sha256=r.sha(r.BENCH),backup=str(P/'bench_small')),indent=2))
launch([sys.executable,str(P/'cache_isa.py')],'final_isa.out')
for suite in ['regression','accuracy','stress','perf','profile']:
 launch([sys.executable,str(P/'small_run.py'),suite],'small_'+suite+'.out')
launch([sys.executable,str(P/'run_native.py')],'native_reference.out')
launch([sys.executable,str(P/'analyze_all.py')],'summary.out')
print('FORMAL_SMALL_CAMPAIGN_PASS',r.stamp(),flush=True)
