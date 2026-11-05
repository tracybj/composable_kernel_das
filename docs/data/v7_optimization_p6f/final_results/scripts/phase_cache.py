"""Finite first phase; stop on any failed subprocess, never silently retry."""
import json,subprocess,sys,time
import runtime as r
resume='--resume' in sys.argv[1:]
for suite in ['regression','cache_native','cache_accuracy','cache_perf']:
    log=r.ROOT/(suite+'.out')
    if log.exists():
        if not resume:raise RuntimeError('Phase output exists; inspect/resume campaign explicitly: '+suite)
        log=r.ROOT/(suite+'_resume_e'+str(r.epoch())+'_'+str(time.time_ns())+'.out')
    cmd=[sys.executable,'-u',str(r.ROOT/'campaign.py'),suite]
    print('PHASE_STAGE',suite,r.stamp(),flush=True)
    start=time.monotonic()
    with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,r.remaining()-300))
    r.append(r.ROOT/'phase_stages.jsonl',dict(suite=suite,command=cmd,rc=p.returncode,at=r.stamp(),seconds=time.monotonic()-start,path=str(log)))
    if p.returncode:raise RuntimeError('STOP '+str(log))
    print('PHASE_STAGE_PASS',suite,flush=True)
print('CACHE_FACTOR_CAMPAIGN_DONE',r.stamp(),flush=True)
