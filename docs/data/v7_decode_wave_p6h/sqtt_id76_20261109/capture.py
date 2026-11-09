"""Recollect ID76 only: paired PMC + SQTT, using the unchanged trial06b binary.

Run through the existing fail-closed GPU4 runtime and global serialization lock.
This does not resume P6g or change any kernel, selector, or historical timing.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import runtime as r
import campaign as c

EXPECTED_SHA = '276fdae07237401fdfb56976c7c6c11faa61418deaf8097e4839607da1fbfc99'
SECTIONS = 'speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state'


def inventory(folder):
    files = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        item = dict(path=str(path), bytes=path.stat().st_size, sha256=r.sha(path))
        if path.suffix == '.out':
            lines = path.read_text().splitlines()
            assert lines and all(re.fullmatch('[0-9a-fA-F]{4}', line) for line in lines), str(path)
            item.update(raw_token_bytes=2 * len(lines), raw_words=len(lines), decoded=False)
        files.append(item)
    assert len([x for x in files if x['path'].endswith('.perf') and x['bytes'] > 0]) == 1
    assert any(x.get('raw_token_bytes', 0) > 0 for x in files), 'No actual SQTT tokens'
    return files


def main():
    r.no_core()
    assert r.sha(c.BENCH) == EXPECTED_SHA, 'The accepted binary changed; do not recollect a different prototype'
    assert not (HERE / 'plan.json').exists(), 'Immutable run: use a new directory for a retry'
    shape = c.shapes()[76]
    assert [shape[k] for k in ['M', 'N', 'K', 'G']] == [1, 2048, 768, 4]
    attempt = uuid.uuid4().hex[:10]
    plan = dict(at=r.stamp(), id=76, shape={k: shape[k] for k in ['M', 'N', 'K', 'G']},
                binary=str(c.BENCH), binary_sha256=EXPECTED_SHA, physical_gpu=4,
                variants={'base': 5008, 'new': 6000}, target_dispatch=18,
                scope='First active shader engine, automatically selected CU; sampled scope, not every CU',
                capture_timing_is_formal=False, p6g_state='remains paused', attempt=attempt,
                script_sha256=r.sha(__file__))
    (HERE / 'plan.json').write_text(json.dumps(plan, indent=2))
    records = []
    try:
        with r.locked():
            r.gate('sqtt76_preflight_' + attempt)
            start_epoch = r.epoch()
            for variant in ['base', 'new']:
                spec = c.bench(shape, variant, 'accuracy', 'sqtt76_' + attempt + '_' + variant + '_acc', repeat=3)
                spec.update(capture_plan=str(HERE / 'plan.json'), binary=str(c.BENCH))
                records.append(r.execute(spec))
                assert r.sha(c.BENCH) == EXPECTED_SHA
            for variant, kernel in [('base', 'GemmPipelineV7SmallMK32'), ('new', 'DecodeNnWaveKernel')]:
                key = 'sqtt76_' + attempt + '_' + variant
                spec = c.bench(shape, variant, 'perf', key)
                command = [arg.replace('--samples=30', '--samples=1').replace('--validate=1', '--validate=0')
                           for arg in spec['command']]
                folder = r.ROOT / 'runs' / key / 'profile'
                spec.update(kind='profile', binary=str(c.BENCH), capture_plan=str(HERE / 'plan.json'),
                            sqtt=True, target_dispatch=18, capture_timing_is_formal=False,
                            logical_tile=[64, 128, 32] if variant == 'base' else [16, 128, 32])
                spec['command'] = ['/opt/rocm-6.3.3/bin/xprof', '--enable-sqtt', '--enable-raw',
                                   '--target-se', '0', '--dispatches', '18', '--kernels', kernel,
                                   '--sections', SECTIONS, '--output-dir', str(folder)] + command
                rec = r.execute(spec)
                rec['capture_files'] = inventory(folder)
                records.append(rec)
                assert r.sha(c.BENCH) == EXPECTED_SHA
                print('SQTT_TOKEN_AUDIT_PASS', variant,
                      sum(x.get('raw_token_bytes', 0) for x in rec['capture_files']), flush=True)
            assert all(rec['epoch'] == start_epoch and rec['ok'] for rec in records), 'No cross-epoch accepted pair'
            transitions = r.kernel_events('sqtt76_pair_complete_' + attempt)
            if transitions:
                r.mark_environment('kernel_service_transition', dict(key=attempt, events=transitions))
                raise RuntimeError('Pair crossed a kernel-service transition; retain and invalidate both profiles')
        result = dict(plan=plan, ok=True, finished_at=r.stamp(), epoch=start_epoch, jobs=records,
                      instruction_latency_decoded_locally=False,
                      note='SQTT tokens captured for XCompute GUI review; capture timings are not formal benchmark results.')
        (HERE / 'result.json').write_text(json.dumps(result, indent=2))
        print('ID76_PAIRED_SQTT_PASS', flush=True)
    except BaseException as exc:
        (HERE / 'failure.json').write_text(json.dumps(dict(at=r.stamp(), error=repr(exc), jobs=records), indent=2))
        raise


if __name__ == '__main__':
    main()
