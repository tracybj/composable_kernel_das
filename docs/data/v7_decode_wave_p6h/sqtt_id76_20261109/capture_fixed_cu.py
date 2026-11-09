"""Use an explicit identical CU for the pair after sparse auto-CU baseline trace.

Keep the initial attempt unchanged. Reuse its just-passed accuracy only if the
binary and environment epoch remain identical. No formal timing is collected.
"""
import json
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture as first
r, c = first.r, first.c


def main():
    r.no_core()
    original = json.loads((HERE / 'result.json').read_text())
    assert original['ok'] and r.sha(c.BENCH) == first.EXPECTED_SHA
    assert not (HERE / 'fixed_cu_result.json').exists()
    shape = c.shapes()[76]
    attempt = uuid.uuid4().hex[:10]
    records = []
    with r.locked():
        r.gate('sqtt76_fixed_cu_preflight_' + attempt)
        assert r.epoch() == original['epoch'], 'Revalidate accuracy after an environment change'
        for variant, kernel in [('base', 'GemmPipelineV7SmallMK32'), ('new', 'DecodeNnWaveKernel')]:
            key = 'sqtt76_cu0_' + attempt + '_' + variant
            spec = c.bench(shape, variant, 'perf', key)
            command = [arg.replace('--samples=30', '--samples=1').replace('--validate=1', '--validate=0')
                       for arg in spec['command']]
            folder = r.ROOT / 'runs' / key / 'profile'
            spec.update(kind='profile', binary=str(c.BENCH), sqtt=True, target_dispatch=18,
                        target_se=0, target_cu_active_index=0,
                        capture_timing_is_formal=False, prior_accuracy_result=str(HERE / 'result.json'),
                        capture_script=str(Path(__file__).resolve()), capture_script_sha256=r.sha(__file__),
                        logical_tile=[64, 128, 32] if variant == 'base' else [16, 128, 32])
            spec['command'] = ['/opt/rocm-6.3.3/bin/xprof', '--enable-sqtt', '--enable-raw',
                               '--target-se', '0', '--target-cu', '0', '--dispatches', '18',
                               '--kernels', kernel, '--sections', first.SECTIONS,
                               '--output-dir', str(folder)] + command
            rec = r.execute(spec)
            rec['capture_files'] = first.inventory(folder)
            records.append(rec)
            raw_bytes = sum(x.get('raw_token_bytes', 0) for x in rec['capture_files'])
            print('FIXED_CU_CAPTURE', variant, 'raw_token_bytes=' + str(raw_bytes), flush=True)
            assert raw_bytes >= 4096, 'Sparse trace: retain but do not deliver as instruction-level evidence'
            assert r.sha(c.BENCH) == first.EXPECTED_SHA
        assert all(x['epoch'] == original['epoch'] and x['ok'] for x in records)
        events = r.kernel_events('sqtt76_fixed_pair_complete_' + attempt)
        if events:
            r.mark_environment('kernel_service_transition', dict(key=attempt, events=events))
            raise RuntimeError('Service transition crossed the pair')
    result = dict(ok=True, at=r.stamp(), epoch=original['epoch'], binary_sha256=first.EXPECTED_SHA,
                  jobs=records, initial_attempt=str(HERE / 'result.json'),
                  accuracy_jobs=[x['key'] for x in original['jobs'] if x['kind'] == 'accuracy'],
                  scope='First active CU in first active SE; compare sampled waves, not the entire device',
                  sparse_auto_cu_baseline='Retained; 800 token bytes, not selected as instruction evidence',
                  capture_timing_is_formal=False, instruction_latency_decoded_locally=False)
    (HERE / 'fixed_cu_result.json').write_text(json.dumps(result, indent=2))
    print('ID76_FIXED_CU_PAIRED_SQTT_PASS', flush=True)


if __name__ == '__main__':
    main()
