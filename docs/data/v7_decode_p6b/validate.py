#!/usr/bin/env python3
"""Audit the complete experimental campaign, without launching GPU work."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'stage_wait'
REPO = Path('/workspace/composable_kernel')


def main():
    jobs = [json.loads(s) for s in (DATA / 'jobs.jsonl').read_text().splitlines()]
    good = {r['key']: r for r in jobs if r['ok']}
    counts = collections.Counter(r['key'].split('_')[0] for r in good.values())
    assert len(jobs) == len(good) == 323, (len(jobs), len(good))
    assert counts == dict(smoke=6, acc=48, reg=5, perf=216, guard=36, prof=12), counts
    comparisons = elements = 0
    for r in good.values():
        log = (Path(r['path']) / 'run.log').read_text()
        if r['key'].startswith(('smoke_', 'acc_')):
            rows = re.findall(r'ACCURACY backend=ck_vs_blas repeat=(\d+) checked=(\d+) failed=(\d+) max_abs=([0-9.eE+-]+)', log)
            assert len(rows) == 3, r['key']
            for repeat, checked, failed, error in rows:
                assert int(failed) == 0 and float(error) == 0, r['key']
                comparisons += 1
                elements += int(checked)
        if r['key'].startswith(('perf_', 'guard_')):
            assert 'SUMMARY result=PASS' in log and r['result'], r['key']
    gates = [json.loads(s) for s in (DATA / 'gate.jsonl').read_text().splitlines()]
    for key in good:
        assert any(g['key'] == key and g['rc'] == 0 and g['hcu4'] == '0.0%' for g in gates), key
    profiles = json.loads((DATA / 'profile_summary.json').read_text())
    assert len(profiles) == 12 and all(p['mmop_count_matches_padded'] for p in profiles)
    compare = (ROOT / 'kernel_compare_final.txt').read_text()
    assert compare.count('SAME  ') == 6 and compare.count('ONLY_B') == 4
    protected = 'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'
    diff = subprocess.check_output(['git', 'diff', '--', protected], cwd=REPO)
    assert hashlib.md5(diff).hexdigest() == '4c7f49a5c7592767d53195a713db6b8a'
    result = dict(valid_jobs=len(good), counts=dict(counts), full_output_comparisons=comparisons,
                  checked_elements=elements, failed=0, max_abs=0, gate_records=len(gates),
                  profile_geometry_dispatches=24, original_v7_symbols_unchanged=6,
                  protected_diff_md5=hashlib.md5(diff).hexdigest())
    (DATA / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('CAMPAIGN_PASS', json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
