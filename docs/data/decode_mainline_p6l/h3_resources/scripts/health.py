"""One independent, bounded recovery. Never retries a failing HIP process."""
import datetime
import json
import time
import uuid
from pathlib import Path
import plan
import runtime as r


def probe_evidence():
    path = Path('/workspace/v7_coverage/p6e/jobs.jsonl')
    for line in path.open():
        x = json.loads(line)
        if x['key'] == 'acc_cache_id76_base':
            assert x['ok'] and x['selected'] == 5000 and len(x['comparisons']) == 3
            assert all(int(c[1]) > 0 and int(c[2]) == 0 for c in x['comparisons'])
            assert plan.sha(r.PROBE) == r.PROBE_SHA == x['benchmark_sha256']
            return dict(binary=str(r.PROBE), sha256=r.PROBE_SHA, prior_key=x['key'],
                        evidence_path=str(path), evidence_record_sha256=plan.digest(x),
                        comparisons=x['comparisons'])
    raise ValueError('Original5000 whole-output evidence missing')


def recover(site):
    r.no_core()
    with r.locked():
        if site.stop.exists(): raise r.QueueStopped('Permanent stop marker; recovery forbidden')
        if not site.mark.exists(): raise ValueError('No environment incident to recover')
        marker = r.read(site.mark)
        total = r.read(site.root/'recovery_total.json', dict(seconds=0, attempts=0))
        begin = time.monotonic()
        incident_time = datetime.datetime.fromisoformat(marker['at']).timestamp()
        elapsed = lambda: max(time.monotonic()-begin, time.time()-incident_time)
        key = 'recovery_'+uuid.uuid4().hex[:16]
        success = False; checks = []; record = dict(key=key, at=r.stamp(), marker=marker)
        try:
            if probe_evidence() != site.manifest['probe']:
                raise ValueError('Known probe/evidence changed')
            consecutive = 0
            for attempt in range(1, 4):
                if site.gpu_remaining() < 210 or total['seconds']+elapsed() >= 5400:
                    raise r.QueueStopped('Recovery/session time limit')
                if attempt > 1:
                    before = time.monotonic(); time.sleep(60)
                    site.events('waits.jsonl', reason='recovery_spacing', key=key, seconds=time.monotonic()-before)
                transitions = site.kernel_events(key+'_observation'+str(attempt))
                healthy, check = site.check(key, attempt)
                check['kernel_service_events'] = transitions; checks.append(check)
                consecutive = consecutive+1 if healthy and not transitions else 0
                if consecutive == 2: break
            if consecutive != 2:
                raise r.QueueStopped('No consecutive healthy pair within3observations')
            # Last observation is also the probe's immediately-prior gate.
            # Do not insert an extra unspaced recovery check.
            remaining = min(300, site.gpu_remaining()-30, 5400-total['seconds']-elapsed())
            if remaining < 30: raise r.QueueStopped('No probe budget')
            assert plan.sha(r.PROBE) == r.PROBE_SHA
            command = [str(r.PROBE), '--dtype=bf16', '--layout=NN', '--m=1', '--n=2048',
                       '--k=768', '--groups=4', '--backend=both', '--blas-lib='+str(plan.BLAS),
                       '--include-args-build=0', '--time=0', '--validate=1', '--correctness-repeats=3']
            folder = site.root/'recovery'/key; folder.mkdir(parents=True, exist_ok=False)
            record.update(command=command, path=str(folder), binary_sha256=r.PROBE_SHA,
                          physical_gpu=4, gate_at=checks[-1]['at'])
            r.write(folder/'command.json', record)
            rc, pid = site.run_program(command, folder, plan.environment({plan.V7:'1'}), remaining)
            out = (folder/'run.log').read_text(errors='replace')
            transitions = site.kernel_events(key+'_probe_post')
            record.update(rc=rc, pid=pid, kernel_service_events=transitions)
            if rc != 0 or transitions: raise r.QueueStopped('Known-correct probe failed/environment changed')
            record.update(r.parse_output(out, 'probe', 5000, repeats=3))
            epoch = site.epoch()+1
            r.write(site.root/'environment_epoch.json', dict(id=epoch, at=r.stamp(), recovery_key=key))
            site.mark.rename(site.root/(key+'_resolved.json'))
            success = True
            print('RECOVERY_PASS epoch='+str(epoch), flush=True)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            duration = elapsed(); total['seconds'] += duration; total['attempts'] += 1
            r.write(site.root/'recovery_total.json', total)
            record.update(finished_at=r.stamp(), checks=checks, ok=success, seconds=duration)
            r.append(site.root/'recovery.jsonl', record)
            if not success or total['seconds'] >= 5400:
                if not site.stop.exists(): r.write(site.stop, dict(at=r.stamp(), reason='RECOVERY_FAILED_OR_90MIN_LIMIT', total=total))
            site.status(state='recovered' if success and not site.stop.exists() else 'GPU_queue_stopped', recovery_key=key)
