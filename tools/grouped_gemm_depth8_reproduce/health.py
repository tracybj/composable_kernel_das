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


def session_probe_definition(benchmark_sha):
    return dict(mode='session_validated_v7',binary=str(plan.BENCH),sha256=benchmark_sha,
                case=dict(ID=-6003000,M=1,N=2048,K=768,G=4,expected=5000,
                          env={plan.V7:'1'},accuracy_repeats=3))


def verified_probe(site):
    """A recovery probe must have genuine prior full-output evidence."""
    definition=site.manifest['probe']
    if definition.get('mode')!='session_validated_v7':
        if probe_evidence()!=definition:
            raise ValueError('Known probe/evidence changed')
        return definition
    frozen_digest=plan.digest({k:v for k,v in site.manifest.items() if k!='plan_sha256'})
    if frozen_digest!=site.manifest['plan_sha256']:
        raise ValueError('Frozen recovery manifest changed')
    site.verify_frozen_files()
    if definition!=session_probe_definition(site.manifest['benchmark_sha256']):
        raise ValueError('Unexpected session recovery control')
    proof=r.read(site.root/'probe_validation.json')
    if not proof or proof.get('plan_sha256')!=site.manifest['plan_sha256']:
        raise ValueError('Recovery control has not been validated in this session')
    record=proof['record'];case=definition['case']
    if (proof['record_sha256']!=plan.digest(record) or not record.get('ok') or
        record.get('rc')!=0 or record.get('purpose')!='recovery_probe_bootstrap' or
        record.get('physical_gpu')!=4 or record.get('kernel_service_events') or
        record.get('binary_sha256')!=definition['sha256'] or
        record.get('benchmark_sha256')!=site.manifest['benchmark_sha256'] or
        record.get('plan_sha256')!=site.manifest['plan_sha256'] or
        record.get('selected')!=5000 or record.get('kind')!='accuracy'):
        raise ValueError('Invalid recovery-control record')
    expected_env={k:v for k,v in plan.environment(case['env']).items()
                  if k.startswith((plan.PREFIX,'HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS'))}
    if record.get('environment')!=expected_env:
        raise ValueError('Recovery-control environment changed')
    journal=site.root/'jobs.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    matches=[row for row in rows if row.get('key')==record['key']]
    if len(matches)!=1 or plan.digest(matches[0])!=proof['record_sha256']:
        raise ValueError('Recovery control does not match its physical job journal')
    log=Path(record['path'])/'run.log'
    if plan.sha(log)!=proof['log_sha256']:
        raise ValueError('Recovery-control log changed')
    parsed=r.parse_output(log.read_text(),'accuracy',5000,repeats=3)
    dims={arg:str(case[k]) for arg,k in [('m','M'),('n','N'),('k','K'),('groups','G')]}
    if (any(parsed['config'].get(k)!=v for k,v in dims.items()) or
        [list(row) for row in parsed['comparisons']]!=record['comparisons'] or
        any(int(row[1])!=8192 for row in parsed['comparisons'])):
        raise ValueError('Recovery control did not validate all M*N*G outputs')
    if site.mark.exists():
        marker=r.read(site.mark)
        if datetime.datetime.fromisoformat(record['finished_at'])>datetime.datetime.fromisoformat(marker['at']):
            raise ValueError('Recovery evidence must precede the environment incident')
    return definition


def ensure_session_probe(site):
    """Called under the shared lock; dry-run never enters this function."""
    if site.manifest.get('probe',{}).get('mode')!='session_validated_v7':return
    proof=site.root/'probe_validation.json'
    if proof.exists():
        try:verified_probe(site)
        except (ValueError,KeyError,OSError) as error:
            site.halt('RECOVERY_PROOF_INVALID',error=str(error))
        return
    case=site.manifest['probe']['case']
    try:
        record=site.execute(dict(logical_key='recovery_probe_bootstrap',purpose='recovery_probe_bootstrap',
                                ID=case['ID'],shape={k:case[k] for k in ['M','N','K','G']},
                                kind='accuracy',round=0,repeats=3,expected=5000,env=case['env'],
                                attempt=uuid.uuid4().hex[:16],command=plan.command(case,'accuracy')))
    except r.EnvironmentFailure as error:
        site.halt('BOOTSTRAP_CONTROL_NOT_VALIDATED',error=str(error))
    r.write(proof,dict(plan_sha256=site.manifest['plan_sha256'],record=record,
                       record_sha256=plan.digest(record),log_sha256=plan.sha(Path(record['path'])/'run.log')))
    verified_probe(site)
    print('SESSION_RECOVERY_PROBE_PASS instance5000 full_output_repeats3',flush=True)


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
            probe=verified_probe(site)
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
            probe_binary=Path(probe.get('binary',r.PROBE))
            probe_sha=probe.get('sha256',r.PROBE_SHA)
            if plan.sha(probe_binary)!=probe_sha:
                raise ValueError('Validated probe binary changed')
            session_probe=probe.get('mode')=='session_validated_v7'
            command = plan.command(probe['case'],'accuracy') if session_probe else [str(probe_binary), '--dtype=bf16', '--layout=NN', '--m=1', '--n=2048',
                       '--k=768', '--groups=4', '--backend=both', '--blas-lib='+str(plan.BLAS),
                       '--include-args-build=0', '--time=0', '--validate=1', '--correctness-repeats=3']
            folder = site.root/'recovery'/key; folder.mkdir(parents=True, exist_ok=False)
            record.update(command=command, path=str(folder), binary_sha256=probe_sha,
                          physical_gpu=4, gate_at=checks[-1]['at'])
            r.write(folder/'command.json', record)
            rc, pid = site.run_program(command, folder, plan.environment({plan.V7:'1'}), remaining)
            out = (folder/'run.log').read_text(errors='replace')
            transitions = site.kernel_events(key+'_probe_post')
            record.update(rc=rc, pid=pid, kernel_service_events=transitions)
            if rc != 0 or transitions: raise r.QueueStopped('Known-correct probe failed/environment changed')
            record.update(r.parse_output(out, 'accuracy' if session_probe else 'probe', 5000, repeats=3))
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
