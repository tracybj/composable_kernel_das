"""Independent decode experiments, GPU4 serial and fail-closed health."""
import datetime, fcntl, hashlib, json, math, os, re, resource, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = Path('/workspace/composable_kernel')
LOCK = Path('/workspace/v7_coverage/p6b/run.lock')
MARK = ROOT / 'environment_recovery_required.json'
STOP = ROOT / 'gpu_stop.json'
BLAS = '/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so'

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def session(): return json.loads((ROOT/'session.json').read_text())
def append(path, data):
    with Path(path).open('a') as f:
        f.write(json.dumps(data,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
def remaining(new=False):
    return float('inf') # New task, no user-provided wall-clock budget; never reads P6g session.
def epoch():
    p=ROOT/'environment_epoch.json'; return json.loads(p.read_text())['id'] if p.exists() else 0
def env(overrides=None):
    # Never inherit a previous experiment's opt-in flags.
    e={k:v for k,v in os.environ.items() if not k.startswith('CK_TILE_GROUPED_GEMM_') and k not in ['LD_PRELOAD','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
    card=str(session()['physical_gpu']); assert card == '4'
    e.update(HIP_VISIBLE_DEVICES=card,CUDA_VISIBLE_DEVICES=card,OMP_NUM_THREADS='4',
             CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1',
             LD_LIBRARY_PATH=str(Path(BLAS).parent)+':'+e.get('LD_LIBRARY_PATH',''))
    if overrides:e.update(overrides)
    assert e['HIP_VISIBLE_DEVICES']==card and e['CUDA_VISIBLE_DEVICES']==card
    return e
def no_core(): resource.setrlimit(resource.RLIMIT_CORE,(0,0))
def locked():
    f=LOCK.open('a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); return f
def check(key,attempt=1):
    card=session()['physical_gpu']; start=time.monotonic()
    try:
        p=subprocess.run(['hy-smi'],env=env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=min(45,max(1,remaining())))
        out=re.sub(r'\x1b\[[0-9;]*m','',p.stdout); rc=p.returncode
    except Exception as ex: out=repr(ex); rc=-1
    rows=[line.split() for line in out.splitlines() if re.match(r'^\s*'+str(card)+r'\s',line)]
    hcu=rows[0][6] if len(rows)==1 and len(rows[0])>6 else None
    mem=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])
    row=dict(at=stamp(),key=key,attempt=attempt,rc=rc,physical_gpu=card,hcu=hcu,mem_available_kib=mem,seconds=time.monotonic()-start,output=out)
    append(ROOT/'gate.jsonl',row)
    if mem<64*1024*1024: STOP.write_text(json.dumps(dict(reason='LOW_HOST_MEMORY',check=row),indent=2)); raise RuntimeError('LOW_HOST_MEMORY; queue stopped')
    healthy=rc==0 and 'MKFD' not in out.upper() and hcu is not None
    if healthy and hcu!='0.0%': STOP.write_text(json.dumps(dict(reason='CARD_BUSY',check=row),indent=2)); raise RuntimeError('CARD_BUSY; queue stopped; never preempt')
    return healthy,row
def mark_environment(reason,record):
    if not MARK.exists():
        payload=dict(record)
        if 'ok' in payload:payload['ok']=False
        MARK.write_text(json.dumps(dict(at=stamp(),epoch=epoch(),reason=reason,record=payload),indent=2))
def kernel_events(key):
    """Track service transitions, including gaps between successful HIP jobs."""
    p=subprocess.run(['dmesg','--color=never'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    cursor=ROOT/'kernel_cursor.json'
    prev=json.loads(cursor.read_text())['last_timestamp'] if cursor.exists() else session()['t0_monotonic']
    latest=prev;events=[]
    if p.returncode==0:
        for line in p.stdout.splitlines():
            m=re.match(r'^\[\s*(\d+\.\d+)\]',line)
            if not m:continue
            t=float(m[1]);latest=max(latest,t)
            if t>prev and re.search(r'MKFD|hymgr.*(?:unregister|register)',line,re.I):events.append(line)
        cursor.write_text(json.dumps(dict(last_timestamp=latest,at=stamp()),indent=2))
    append(ROOT/'kernel_checks.jsonl',dict(at=stamp(),key=key,rc=p.returncode,from_timestamp=prev,to_timestamp=latest,events=events,error=p.stdout[:300] if p.returncode else None))
    return events
def gate(key):
    if STOP.exists(): raise RuntimeError('GPU_QUEUE_STOPPED '+str(STOP))
    if MARK.exists(): raise RuntimeError('ENVIRONMENT_RECOVERY_REQUIRED; separate fresh process only')
    if remaining()<60: raise RuntimeError('DEADLINE')
    transitions=kernel_events(key+'_pre')
    if transitions: mark_environment('kernel_service_transition',dict(key=key,events=transitions)); raise RuntimeError('ENVIRONMENT_RECOVERY_REQUIRED')
    ok,row=check(key)
    if not ok: mark_environment('pre_job_gate',row); raise RuntimeError('ENVIRONMENT_RECOVERY_REQUIRED')
    return row
def execute(spec):
    """Caller owns global lock. One fresh HIP process and immutable physical key."""
    if spec.get('kind') in ['perf','diagnostic_perf']:
        launches=int(next((a.split('=',1)[1] for a in spec['command'] if a.startswith('--launches-per-sample=')),'1'))
        if spec['kind']=='perf': assert launches==1,'Formal timing must use one launch per sample'
        else: assert launches in [1,20] and launches==spec['launches_per_sample']
    key=spec['key']; assert re.fullmatch(r'[A-Za-z0-9_.-]+',key)
    folder=ROOT/'runs'/key
    if folder.exists(): raise RuntimeError('Physical key already exists; never overwrite '+key)
    gate_row=gate(key)
    folder.mkdir(parents=True)
    e=env(spec.get('env')); cmd=spec['command']
    binary=spec.get('binary',cmd[0])
    rec=dict(spec,at=stamp(),epoch=epoch(),physical_gpu=session()['physical_gpu'],path=str(folder),binary_sha256=sha(binary),script_sha256=sha(__file__),gate_at=gate_row['at'],environment={k:v for k,v in e.items() if k.startswith(('CK_TILE_GROUPED_GEMM_','HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'))})
    versions=ROOT/'scripts';versions.mkdir(exist_ok=True);rec['source_scripts_sha256']={}
    for name in ['runtime.py','health.py','campaign.py','native_replay.py']:
        source=ROOT/name
        if source.exists():
            digest=sha(source);rec['source_scripts_sha256'][name]=digest
            snapshot=versions/(digest+'_'+name)
            if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
    (folder/'command.json').write_text(json.dumps(rec,indent=2))
    append(ROOT/'started.jsonl',rec)
    print('START',key,flush=True); start=time.monotonic()
    with (folder/'run.log').open('w') as f:
        p=subprocess.Popen(cmd,env=e,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,preexec_fn=no_core)
        rec['pid']=p.pid; (folder/'live.json').write_text(json.dumps(dict(pid=p.pid,at=stamp()),indent=2))
        (ROOT/'live_status.json').write_text(json.dumps(dict(at=stamp(),state='GPU_job_running',key=key,pid=p.pid,physical_gpu=rec['physical_gpu'],epoch=rec['epoch'],log=str(folder/'run.log')),indent=2))
        timeout=min(spec.get('timeout',1800),max(1,remaining()-30))
        try: rec['rc']=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            rec.update(rc=None,live_pid=p.pid,error='OBSERVATION_TIMEOUT; process left running; no new GPU submissions')
            STOP.write_text(json.dumps(rec,indent=2))
    rec.update(seconds=time.monotonic()-start,finished_at=stamp())
    out=(folder/'run.log').read_text(errors='replace')
    rec['ok']=rec['rc']==0 and 'MKFD' not in out.upper()
    transitions=kernel_events(key+'_post');rec['kernel_service_events']=transitions
    if transitions:mark_environment('kernel_service_transition',rec);rec['ok']=False
    if 'MKFD' in out.upper() or re.search(r'invalid device ordinal|hsa.*error|device.*lost',out,re.I):
        mark_environment('job_environment_suspected',rec); rec['ok']=False
    if spec.get('expected') is not None:
        selected=re.search(r'SELECT instance_id=(\d+)',out)
        rec['selected']=int(selected[1]) if selected else None
        rec['ok'] &= rec['selected']==spec['expected'] and 'SUMMARY result=PASS' in out
    if spec.get('expected_not') is not None:
        selected=re.search(r'SELECT instance_id=(\d+)',out)
        rec['selected']=int(selected[1]) if selected else None
        rec['ok'] &= rec['selected'] is not None and rec['selected'] not in spec['expected_not'] and 'SUMMARY result=PASS' in out
    if spec.get('kind')=='accuracy':
        checks=re.findall(r'ACCURACY backend=ck_vs_blas repeat=(\d+) checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)',out)
        wanted=int(next((arg.split('=',1)[1] for arg in cmd if arg.startswith('--correctness-repeats=')),'1'))
        rec['comparisons']=checks; rec['ok'] &= len(checks)==wanted and all(int(x[1])>0 and int(x[2])==0 for x in checks)
        for backend in ['ck_redzone','blas_redzone']:
            guards=re.findall(r'ACCURACY backend='+backend+r' repeat=(\d+) checked=(\d+) failed=(\d+)',out)
            rec['ok'] &= len(guards)==wanted and all(int(x[1])==32 and int(x[2])==0 for x in guards)

    if spec.get('kind') in ['perf','diagnostic_perf']:
        m=re.search(r'PERF backend=ck_device_args (.*)',out)
        rec['result']={k:float(v) for k,v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)',m[1])} if m else None
        rec['ok'] &= bool(m) and bool(rec['result']) and set(rec['result'])=={'median_ms','min_ms','max_ms','cv'} and rec['result']['median_ms']>0 and bool(re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)',out))
        if rec['result']:
            stats=rec['result'];rec['ok'] &= all(math.isfinite(v) for v in stats.values()) and 0<stats['min_ms']<=stats['median_ms']<=stats['max_ms'] and stats['cv']>=0

    if spec.get('kind')=='native_perf':
        lines=[line[7:] for line in out.splitlines() if line.startswith('RESULT ')]
        rec['result']=json.loads(lines[-1]) if lines else None
        rec['ok'] &= bool(lines)
    if spec.get('kind')=='profile':
        rec['perf_files']=[str(p) for p in (folder/'profile').glob('*.perf')]
        rec['ok'] &= len(rec['perf_files'])==1
    if spec.get('kind')=='regression':
        suffix=spec['regression_kind']
        if suffix=='selector_test': rec['ok'] &= 'SELECTOR_TABLE_PASS' in out
        elif suffix=='device_args_query_test':
            rec['ok'] &= 'DEVICE_ARGS_QUERY_PASS' in out and 'QUERY_V7_BYTE_RANGE_PASS cases=7' in out
        elif suffix=='device_args_smoke':
            m=re.search(r'DEVICE_ARGS_SMOKE arch=(\S+) passed=(\d+) cases=(\d+) negative=(\d+).*result=PASS',out)
            rec['ok'] &= bool(m) and int(m[2])==int(m[3]) and int(m[3])>0 and m[4]=='1'
    append(ROOT/'jobs.jsonl',rec); print('DONE',key,'ok='+str(rec['ok']),'sec='+str(round(rec['seconds'],2)),flush=True)
    (ROOT/'live_status.json').write_text(json.dumps(dict(at=stamp(),state='GPU_job_passed' if rec['ok'] else 'GPU_queue_stopped',key=key,rc=rec['rc'],physical_gpu=rec['physical_gpu'],epoch=rec['epoch'],log=str(folder/'run.log')),indent=2))
    if not rec['ok']: raise RuntimeError('STOP inspect '+str(folder/'run.log'))
    return rec

def preflight():
    no_core()
    with locked():
        gate('initial_preflight')
        p=subprocess.run(['dmesg','--time-format','iso'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        lines=p.stdout.splitlines(); (ROOT/'kernel_preflight.log').write_text('\n'.join(lines[-4000:])+'\n')
        print('PREFLIGHT_PASS physical_gpu='+str(session()['physical_gpu'])+' epoch='+str(epoch()),flush=True)
if __name__=='__main__': preflight()
