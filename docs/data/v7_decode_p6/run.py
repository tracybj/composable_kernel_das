#!/usr/bin/env python3
"""Card-4 serial gate, interleaved measurements, durable per-job resume."""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BENCH = Path('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
BASELINE = Path('/workspace/Grouped_Geem/blaslt_tuning_tools/results/bf16_nn_compare_20261027_191315/groupgemm')
XPROF = '/opt/rocm-6.3.3/bin/xprof'
CASES = [(76, 1, 2048, 768, 4), (681, 1, 1536, 2048, 4),
         (633, 4, 1024, 2048, 8), (358, 8, 6144, 2048, 32)]
GC = 'speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state'
ENV = dict(os.environ, HIP_VISIBLE_DEVICES='4', CUDA_VISIBLE_DEVICES='4',
           CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='1', OMP_NUM_THREADS='4',
           PYTHONUNBUFFERED='1')


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def append(path, record):
    with path.open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def gate(key):
    for attempt in range(1, 4):
        completed = subprocess.run(['hy-smi'], env=ENV, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=45)
        output = re.sub(r'\x1b\[[0-9;]*m', '', completed.stdout)
        rows = [line.split() for line in output.splitlines() if re.match(r'^\s*4\s', line)]
        pct = rows[0][6] if len(rows) == 1 and len(rows[0]) > 6 else None
        append(ROOT / 'gate.jsonl', dict(at=stamp(), key=key, attempt=attempt,
               rc=completed.returncode, hcu4=pct, output=output))
        if completed.returncode == 0 and 'MKFD' not in output and pct is not None:
            if pct != '0.0%':
                raise RuntimeError(f'Card 4 busy: {pct}; stopping without changing card')
            return
        print(f'GATE_RETRY key={key} attempt={attempt} hcu4={pct}', flush=True)
        if attempt < 3:
            time.sleep(60)
    raise RuntimeError('Card-4 gate failed three consecutive times; stopping')


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['timing', 'profile'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (BASELINE / 'results.csv').open(encoding='utf-8-sig') as f:
            frozen = {int(r['ID']): r for r in csv.DictReader(f)}
        manifest = ROOT / 'manifest.json'
        if not manifest.exists():
            paths = [BENCH, Path(XPROF), ROOT/'run.py', ROOT/'blas_replay.py',
                     Path('/workspace/Grouped_Geem/blaslt_tuning_tools/build/libgroupgemm_tune.so')]
            manifest.write_text(json.dumps(dict(at=stamp(), cases=CASES, gpu=4,
                sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), indent=2))
        records = ROOT / 'jobs.jsonl'
        completed_keys = {r['key'] for line in records.read_text().splitlines()
                          if (r := json.loads(line)).get('ok')} if records.exists() else set()
        suites = [('timing', 3)] if args.mode == 'timing' else [('gc', 3), ('tlb', 1)]
        for suite, rounds in suites:
            for round_number in range(1, rounds+1):
                for case_id, m, n, k, g in CASES:
                    for backend in ['ck', 'blas']:
                        key = f'{suite}_r{round_number}_id{case_id}_{backend}'
                        if key in completed_keys:
                            print('SKIP ' + key, flush=True)
                            continue
                        gate(key)
                        attempts = ROOT / 'runs' / key
                        attempts.mkdir(parents=True, exist_ok=True)
                        jobdir = attempts / str(len(list(attempts.iterdir()))+1)
                        jobdir.mkdir()
                        timing = suite == 'timing'
                        warmup, samples = (10, 30) if timing else (1, 1)
                        if backend == 'ck':
                            command = [str(BENCH), '--dtype=bf16', '--layout=NN',
                                f'--m={m}', f'--n={n}', f'--k={k}', f'--groups={g}',
                                '--backend=ck', '--time=1', f'--warmup={warmup}',
                                f'--samples={samples}', '--include-args-build=0',
                                '--correctness-repeats=1', f'--validate={int(timing)}']
                            kernel = 'V7Uniform'
                        else:
                            command = [sys.executable, str(ROOT/'blas_replay.py'),
                                f'--id={case_id}', f'--warmup={warmup}', f'--samples={samples}',
                                '--repeat=1', f'--validate={int(timing)}']
                            kernel = frozen[case_id]['kernel_name']
                        if not timing:
                            command = [XPROF, '--kernels', kernel, '--sections',
                                GC if suite == 'gc' else 'utcl2_statistics,umc_statistics',
                                '--output-dir', str(jobdir/'profile')] + command
                        record = dict(key=key, at=stamp(), suite=suite, round=round_number,
                                      id=case_id, backend=backend, command=command,
                                      path=str(jobdir), gpu=4)
                        job_env = ENV.copy()
                        if backend == 'blas':
                            # Match the original tuning controller: PyTorch otherwise
                            # loads a different hipBLASLt before the bridge is opened.
                            library = '/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so.0.10'
                            job_env['LD_PRELOAD'] = library
                            job_env['LD_LIBRARY_PATH'] = str(Path(library).parent) + ':' + job_env.get('LD_LIBRARY_PATH', '')
                        record['script_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [ROOT/'run.py', ROOT/'blas_replay.py']}
                        (jobdir/'command.json').write_text(json.dumps(record, indent=2))
                        print('START ' + key, flush=True)
                        start = time.monotonic()
                        try:
                            with (jobdir/'run.log').open('w') as log:
                                proc = subprocess.run(command, env=job_env, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=900)
                            record['rc'] = proc.returncode
                            output = (jobdir/'run.log').read_text(errors='replace')
                            record['ok'] = proc.returncode == 0
                            if not timing:
                                record['perf_files'] = [str(p) for p in (jobdir/'profile').glob('*.perf')]
                                record['ok'] &= bool(record['perf_files'])
                            elif backend == 'blas':
                                lines = [line for line in output.splitlines() if line.startswith('RESULT ')]
                                record['result'] = json.loads(lines[-1][7:]) if lines else None
                                record['ok'] &= bool(lines)
                            else:
                                perf = re.search(r'PERF backend=ck(?:_device_args)? .*?median_ms=([0-9.eE+-]+)', output)
                                selected = re.search(r'instance_id=(\d+)', output)
                                record['result'] = dict(median_ms=float(perf[1]), instance_id=int(selected[1])) if perf and selected else None
                                if record['result']:
                                    for field in ['mean_ms', 'min_ms', 'max_ms', 'cv']:
                                        value = re.search(r'\b' + field + r'=([0-9.eE+-]+)', output)
                                        if value:
                                            record['result'][field] = float(value[1])
                                record['ok'] &= bool(perf and selected and int(selected[1]) == 5000
                                    and re.search(r'ACCURACY backend=ck .*?failed=0(?:\s|/)', output))
                        except subprocess.TimeoutExpired:
                            record.update(rc=124, ok=False)
                        record['elapsed_seconds'] = time.monotonic()-start
                        append(records, record)
                        print(f'DONE {key} rc={record["rc"]} ok={record["ok"]} sec={record["elapsed_seconds"]:.1f}', flush=True)
                        if not record['ok']:
                            raise RuntimeError(f'Failed {key}; evidence preserved in {jobdir}')
        print('ALL_DONE ' + args.mode, flush=True)


if __name__ == '__main__':
    main()
