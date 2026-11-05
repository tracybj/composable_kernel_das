"""Explicit bounded recovery only; never called implicitly by the job runner."""
import datetime,json,re,subprocess,time
from pathlib import Path
import runtime as r

def recover():
    r.no_core()
    with r.locked():
        if r.STOP.exists(): raise RuntimeError('GPU queue stopped; do not bypass stop marker')
        if not r.MARK.exists(): raise RuntimeError('No environment failure to recover')
        marker=json.loads(r.MARK.read_text()); start=time.monotonic()
        failure_epoch=datetime.datetime.fromisoformat(marker['at']).timestamp()
        def observed_seconds():return max(time.monotonic()-start,time.time()-failure_epoch)
        total_path=r.ROOT/'recovery_total.json'
        total=json.loads(total_path.read_text()) if total_path.exists() else dict(seconds=0,attempts=0)
        key='recovery_'+str(time.time_ns());success=False;checks=[]
        try:
            consecutive=0
            for attempt in range(1,4):
                if r.remaining()<360 or total['seconds']+observed_seconds()>=5400: raise RuntimeError('RECOVERY_BUDGET')
                if attempt>1:
                    wait_start=time.monotonic();time.sleep(60)
                    r.append(r.ROOT/'waits.jsonl',dict(at=r.stamp(),reason='recovery_spacing',seconds=time.monotonic()-wait_start,key=key))
                ok,row=r.check(key,attempt);checks.append(row);consecutive=consecutive+1 if ok else 0
                if consecutive==2:break
            if consecutive!=2:raise RuntimeError('No two consecutive healthy observations within three checks')
            old=Path('/workspace/v7_coverage/p6e')
            prior=next(json.loads(x) for x in (old/'jobs.jsonl').read_text().splitlines() if json.loads(x)['key']=='acc_cache_id76_base' and json.loads(x).get('ok'))
            binary=old/'bench_cache';assert r.sha(binary)==prior['benchmark_sha256']
            cmd=[str(binary),'--dtype=bf16','--layout=NN','--m=1','--n=2048','--k=768','--groups=4','--backend=both','--blas-lib='+r.BLAS,'--include-args-build=0','--time=0','--validate=1','--correctness-repeats=3']
            # Consume and save the recovery window's service history before the
            # known-good probe. Any subsequent transition invalidates recovery.
            r.kernel_events(key+'_recovery_window')
            # Fresh check immediately before probe. Recovery marker stays active.
            ok,row=r.check(key+'_probe_gate');checks.append(row)
            if not ok:raise RuntimeError('Probe preflight failed')
            folder=r.ROOT/'runs'/key;folder.mkdir(parents=True)
            with (folder/'run.log').open('w') as f:
                p=subprocess.Popen(cmd,env=r.env(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                (folder/'live.json').write_text(json.dumps(dict(pid=p.pid,at=r.stamp(),command=cmd),indent=2))
                try:p.wait(timeout=min(300,max(1,r.remaining()-30)))
                except subprocess.TimeoutExpired:
                    r.STOP.write_text(json.dumps(dict(reason='RECOVERY_PROBE_OBSERVATION_TIMEOUT',pid=p.pid,path=str(folder),at=r.stamp()),indent=2))
                    raise RuntimeError('Probe remains in flight; no new GPU submissions, do not reset/kill other processes')
            out=(folder/'run.log').read_text();checks_output=re.findall(r'ACCURACY backend=ck_vs_blas .*?failed=(\d+)',out)
            transitions=r.kernel_events(key+'_probe_post')
            success=p.returncode==0 and not transitions and 'MKFD' not in out.upper() and 'SELECT instance_id=5000' in out and 'SUMMARY result=PASS' in out and len(checks_output)==3 and all(x=='0' for x in checks_output)
            r.append(r.ROOT/'recovery.jsonl',dict(at=r.stamp(),key=key,marker=marker,checks=checks,command=cmd,binary_sha256=r.sha(binary),prior_full_output_key=prior['key'],rc=p.returncode,ok=success,path=str(folder)))
            if not success:raise RuntimeError('Known-correct original5000 probe failed')
            new_epoch=r.epoch()+1
            (r.ROOT/'environment_epoch.json').write_text(json.dumps(dict(id=new_epoch,started_at=r.stamp(),recovery_key=key),indent=2))
            r.MARK.rename(r.ROOT/(key+'_resolved.json'))
            print('RECOVERY_OBSERVATION_PASS epoch='+str(new_epoch)+'; rerun affected full three-round A/B with new keys',flush=True)
        finally:
            total['seconds']+=observed_seconds();total['attempts']+=1;total_path.write_text(json.dumps(total,indent=2))
            if not success or total['seconds']>=5400:r.STOP.write_text(json.dumps(dict(at=r.stamp(),reason='RECOVERY_FAILED_OR_LIMIT',total=total),indent=2))
if __name__=='__main__':recover()
