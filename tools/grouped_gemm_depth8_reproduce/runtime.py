"""Fail-closed serial GPU execution with immutable physical records."""
import contextlib
import datetime
import fcntl
import json
import math
import os
import re
import resource
import subprocess
import time
import uuid
from pathlib import Path
import plan

LOCK = Path('/workspace/v7_coverage/p6b/run.lock')
PROBE = Path('/workspace/v7_coverage/p6e/bench_cache')
PROBE_SHA = '1052b95c681371b0713c357ade4e3ede659df08dd4d763e3cf6023e9ab5d0d2d'


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, separators=(',', ':'))+'\n'); f.flush(); os.fsync(f.fileno())


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@contextlib.contextmanager
def locked():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class QueueStopped(RuntimeError): pass
class EnvironmentFailure(RuntimeError): pass


def parse_output(text, kind, expected=None, repeats=1, regression=None):
    if 'MKFD' in text.upper(): raise ValueError('MKFD in job output')
    selected = re.search(r'SELECT instance_id=(\d+)', text)
    selected = int(selected[1]) if selected else None
    if expected is not None and selected != expected:
        raise ValueError(f'Instance mismatch: expected{expected}, actual{selected}')
    result = dict(selected=selected)
    config_line = re.search(r'^CONFIG (.*)$',text,re.M)
    config = dict(re.findall(r'(\w+)=(\S+)',config_line[1])) if config_line else {}
    result['config'] = config
    if kind in ['accuracy','perf']:
        wanted=dict(arch='gfx936',dtype='bf16',layout='NN',include_args_build='0',
                    launches_per_sample='1',variable_axis='none',num_cu='0',blas_streams='1')
        if kind=='perf': wanted.update(timing='on',warmup='10',samples='30',output_redzone_elements='0')
        else: wanted.update(timing='off',output_redzone_elements='16')
        if any(config.get(k)!=v for k,v in wanted.items()):
            raise ValueError('Benchmark CONFIG violates frozen protocol')
    if kind in ['accuracy', 'perf', 'probe'] and 'SUMMARY result=PASS' not in text:
        raise ValueError('Missing benchmark PASS')
    if kind in ['accuracy', 'probe']:
        comparisons = re.findall(r'ACCURACY backend=ck_vs_blas repeat=(\d+) checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)', text)
        if len(comparisons) != repeats or [int(x[0]) for x in comparisons]!=list(range(repeats)) or any(int(x[1]) <= 0 or int(x[2]) for x in comparisons):
            raise ValueError('Missing/failed whole-output comparison')
        if any(not math.isfinite(float(x[3])) or float(x[3])<0 for x in comparisons):
            raise ValueError('Nonfinite/invalid full-output comparison error')
        for backend in ['ck_guard', 'blas_guard']:
            guards = re.findall(r'ACCURACY backend='+backend+r' repeat=(\d+) checked=(\d+) failed=(\d+)', text)
            if len(guards) != repeats or any(int(x[2]) for x in guards):
                raise ValueError('Missing/failed '+backend)
        result['comparisons'] = comparisons
        if kind == 'accuracy':
            for backend in ['ck_redzone', 'blas_redzone']:
                guards = re.findall(r'ACCURACY backend='+backend+r' repeat=(\d+) checked=(\d+) failed=(\d+)', text)
                if len(guards) != repeats or any(int(x[1]) != 32 or int(x[2]) for x in guards):
                    raise ValueError('Missing/failed allocation '+backend)
    if kind == 'perf':
        match = re.search(r'PERF backend=ck_device_args (.*)', text)
        stats = {k: float(v) for k, v in re.findall(r'(median_ms|min_ms|max_ms|cv)=([0-9.eE+-]+)', match[1])} if match else {}
        if set(stats) != {'median_ms', 'min_ms', 'max_ms', 'cv'} or not all(math.isfinite(v) for v in stats.values()):
            raise ValueError('Missing/nonfinite timing stats')
        if not 0 < stats['min_ms'] <= stats['median_ms'] <= stats['max_ms'] or stats['cv'] < 0:
            raise ValueError('Invalid timing stats')
        if not re.search(r'ACCURACY backend=ck repeat=0 .*?failed=0(?:\s|$)', text):
            raise ValueError('Missing timing-job CPU/guard validation')
        if 'launches_per_sample=1' not in text:
            raise ValueError('Timing config must explicitly report one launch/sample')
        if 'output_redzone_elements=0' not in text:
            raise ValueError('Formal timing must keep validation-only redzones disabled')
        result['result'] = stats
    if kind == 'regression':
        if regression == 'selector_test' and 'SELECTOR_TABLE_PASS' not in text:
            raise ValueError('Selector regression failed')
        if regression == 'decode_depth8_selector_test' and 'DEPTH8_SELECTOR_PASS' not in text:
            raise ValueError('Depth8 selector regression failed')
        if regression == 'device_args_query_test' and not all(s in text for s in ['DEVICE_ARGS_QUERY_PASS', 'QUERY_V7_BYTE_RANGE_PASS cases=7']):
            raise ValueError('Query/byte-range regression failed')
        if regression == 'device_args_smoke':
            m = re.search(r'DEVICE_ARGS_SMOKE arch=(\S+) passed=(\d+) cases=(\d+) negative=(\d+).*result=PASS', text)
            if not m or int(m[2]) != int(m[3]) or int(m[3]) <= 0 or m[4] != '1':
                raise ValueError('Smoke regression failed')
    return result


class Site:
    def __init__(self, output):
        self.root = Path(output).resolve()
        self.session = read(self.root/'session.json')
        self.manifest = read(self.root/'frozen_plan.json')
        if self.manifest and self.manifest.get('runtime_paths'):
            paths = self.manifest['runtime_paths']
            plan.configure_paths(paths['blas_lib'], paths['shape_files'])
        self.mark = self.root/'environment_recovery_required.json'
        self.stop = self.root/'gpu_stop.json'
        self._identities = {}

    def remaining(self): return self.session['deadline_epoch']-time.time()
    def gpu_remaining(self):
        return min(self.session.get('gpu_deadline_epoch',self.session['deadline_epoch']-7200),
                   self.session['deadline_epoch']-7200)-time.time()
    def epoch(self): return read(self.root/'environment_epoch.json', {'id': 0})['id']
    def events(self, name, **extra): append(self.root/name, dict(at=stamp(), **extra))
    def status(self, **fields): write(self.root/'live_status.json', dict(at=stamp(), **fields))

    def halt(self, reason, **fields):
        write(self.stop, dict(at=stamp(), reason=reason, **fields))
        raise QueueStopped(reason)

    def marker(self, reason, record):
        if not self.mark.exists(): write(self.mark, dict(at=stamp(), epoch=self.epoch(), reason=reason, record=record))

    def identity(self, path, expected):
        path = Path(path); stat = path.stat()
        token = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._identities.get(str(path)) != token:
            if plan.sha(path) != expected: self.halt('BINARY_OR_DEPENDENCY_CHANGED', path=str(path))
            self._identities[str(path)] = token

    def kernel_events(self, key):
        p = subprocess.run(['dmesg', '--color=never'], capture_output=True, text=True, timeout=30)
        previous = read(self.root/'kernel_cursor.json', {'last_timestamp': self.session['t0_monotonic']})['last_timestamp']
        latest = previous; events = []
        if p.returncode == 0:
            for line in p.stdout.splitlines():
                m = re.match(r'^\[\s*(\d+\.\d+)\]', line)
                if not m: continue
                t = float(m[1]); latest = max(latest, t)
                if t > previous and re.search(r'MKFD|hymgr.*(?:unregister|register)', line, re.I): events.append(line)
            write(self.root/'kernel_cursor.json', dict(at=stamp(), last_timestamp=latest))
        self.events('kernel_checks.jsonl', key=key, rc=p.returncode, from_timestamp=previous,
                    to_timestamp=latest, events=events, error=p.stderr[:300] if p.returncode else None)
        if p.returncode:
            self.halt('KERNEL_LOG_UNREADABLE',key=key,error=p.stderr[:300])
        return events

    def check(self, key, attempt=1):
        start = time.monotonic()
        try:
            p = subprocess.run(['hy-smi'], env=plan.environment({}), capture_output=True, text=True,
                               timeout=min(45, max(1, self.remaining())))
            out = re.sub(r'\x1b\[[0-9;]*m', '', p.stdout+p.stderr); rc = p.returncode
        except (subprocess.TimeoutExpired, OSError) as ex: out = str(ex); rc = -1
        lines = [x.split() for x in out.splitlines() if re.match(r'^\s*4\s', x)]
        hcu = lines[0][6] if len(lines) == 1 and len(lines[0]) > 6 else None
        memory = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1])
        record = dict(at=stamp(), key=key, attempt=attempt, rc=rc, output=out, physical_gpu=4,
                      hcu=hcu, mem_available_kib=memory, seconds=time.monotonic()-start)
        self.events('gates.jsonl', **{k:v for k,v in record.items() if k != 'at'})
        if memory < 64*1024*1024: self.halt('LOW_HOST_MEMORY', check=record)
        if rc == 0 and 'MKFD' not in out.upper() and hcu is not None and hcu != '0.0%':
            self.halt('CARD_BUSY', check=record)
        return rc == 0 and 'MKFD' not in out.upper() and hcu == '0.0%', record

    def gate(self, key):
        if self.stop.exists(): raise QueueStopped('Existing stop marker; no new GPU submissions')
        if self.mark.exists(): raise EnvironmentFailure('Separate-process recovery required')
        if self.gpu_remaining() < 60: self.halt('GPU_SUBMISSION_DEADLINE')
        transitions = self.kernel_events(key+'_pre')
        if transitions:
            self.marker('kernel_service_transition', dict(key=key, events=transitions))
            raise EnvironmentFailure('Pre-job kernel service transition')
        healthy, record = self.check(key)
        if not healthy:
            self.marker('pre_job_gate', record)
            raise EnvironmentFailure('Pre-job health failure')
        return record

    def run_program(self, cmd, folder, env, timeout):
        with (folder/'run.log').open('w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, preexec_fn=no_core)
            write(folder/'live.json', dict(at=stamp(), pid=p.pid, command=cmd))
            try: rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                write(self.stop, dict(at=stamp(), reason='OBSERVATION_TIMEOUT_GPU_PROCESS_REMAINS', pid=p.pid, path=str(folder)))
                return None, p.pid
        return rc, p.pid

    def verify_frozen_files(self):
        self.identity(plan.BENCH, self.manifest['benchmark_sha256'])
        self.identity(plan.BLAS, self.manifest['blas_sha256'])
        if self.manifest.get('selector_sha256'):
            self.identity(plan.SELECTOR,self.manifest['selector_sha256'])
        for name,sha in self.manifest.get('runner_sha256',{}).items():
            self.identity(Path(__file__).parent/name,sha)
        for name,sha in self.manifest.get('worktree_source_sha256',{}).items():
            self.identity(plan.REPO/name,sha)
        for name,sha in self.manifest.get('dependencies',{}).get('sha256',{}).items():
            self.identity(name,sha)

    def execute(self, spec):
        """The caller owns LOCK. Every invocation creates one fresh process."""
        key = f"e{self.epoch()}_{spec['logical_key']}_{uuid.uuid4().hex[:10]}"
        gate = self.gate(key)
        self.verify_frozen_files()
        if spec.get('expected_binary_sha256'):
            self.identity(spec['command'][0],spec['expected_binary_sha256'])
        folder = self.root/'jobs'/key; folder.mkdir(parents=True, exist_ok=False)
        env = plan.environment(spec.get('env', {}))
        source_versions = {}
        for source in sorted(Path(__file__).parent.glob('*.py')):
            sha = plan.sha(source); target = self.root/'source_versions'/(sha+'_'+source.name)
            target.parent.mkdir(exist_ok=True)
            if not target.exists(): target.write_bytes(source.read_bytes())
            source_versions[source.name] = sha
        record = dict(spec, key=key, at=stamp(), epoch=self.epoch(), physical_gpu=4, gate_at=gate['at'],
                      path=str(folder), binary_sha256=plan.sha(spec['command'][0]),
                      benchmark_sha256=self.manifest['benchmark_sha256'], plan_sha256=self.manifest['plan_sha256'],
                      environment={k:v for k,v in env.items() if k.startswith((plan.PREFIX, 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS'))},
                      script_sha256=source_versions)
        write(folder/'command.json', record); self.events('started.jsonl', record=record)
        self.status(state='GPU_job_running', key=key, path=str(folder))
        start = time.monotonic(); print('START', key, flush=True)
        rc, pid = self.run_program(spec['command'], folder, env, min(1800, max(1, self.gpu_remaining()-30)))
        record.update(rc=rc, pid=pid, seconds=time.monotonic()-start, finished_at=stamp(), ok=False)
        out = (folder/'run.log').read_text(errors='replace')
        try:
            transitions = self.kernel_events(key+'_post')
        except QueueStopped:
            record['parse_error'] = 'Post-job kernel log unreadable'
            record['kernel_service_events'] = []
            append(self.root/'jobs.jsonl',record)
            raise
        record['kernel_service_events'] = transitions
        suspected = bool(transitions) or 'MKFD' in out.upper() or bool(re.search(r'invalid device ordinal|hsa.*error|device.*lost', out, re.I))
        if suspected: self.marker('job_environment_suspected', record)
        try:
            record.update(parse_output(out, spec['kind'], spec.get('expected'), spec.get('repeats', 1), spec.get('regression_kind')))
            if spec.get('shape'):
                expected_shape={arg:str(spec['shape'][k]) for arg,k in [('m','M'),('n','N'),('k','K'),('groups','G')]}
                if any(record['config'].get(k)!=v for k,v in expected_shape.items()):
                    raise ValueError('Actual CONFIG shape differs from frozen row')
                if spec['kind']=='accuracy':
                    elements=spec['shape']['M']*spec['shape']['N']*spec['shape']['G']
                    if any(int(x[1])!=elements for x in record['comparisons']):
                        raise ValueError('Full-output comparison did not cover M*N*G')
            record['ok'] = rc == 0 and not suspected
        except ValueError as ex: record['parse_error'] = str(ex)
        append(self.root/'jobs.jsonl', record)
        self.status(state='GPU_job_passed' if record['ok'] else 'GPU_queue_stopped', key=key, rc=rc)
        print('DONE', key, 'ok='+str(record['ok']), 'sec='+str(round(record['seconds'], 2)), flush=True)
        if rc is None: raise QueueStopped('GPU process remains in flight; no new submissions')
        if suspected: raise EnvironmentFailure('Post-job environment event')
        if not record['ok']: self.halt('JOB_VALIDATION_FAILURE', key=key, log=str(folder/'run.log'))
        return record
