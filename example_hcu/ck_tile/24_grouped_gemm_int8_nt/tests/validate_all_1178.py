"""Frozen-library acceptance through the additive INT8 cached-weight API.

CPU contract and archive helpers for the original 1178-shape data.
The historical INT32 GPU runner lives in exp/int8_nt.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import statistics

E = Path(__file__).resolve().parents[1]
ROOT = E.parents[2]
BUILD = Path(os.environ.get('CK_INT8_CACHED_BUILD', E / 'build/cached'))
FIELDS = [
    'ID',
    'phase',
    'M',
    'N',
    'K',
    'G',
    'dtype',
    'layout',
    'route',
    'legacy_instance',
    'CK_ms',
    'CK_TOPS',
    'BLAS_target_ms',
    'BLAS_target_TOPS',
    'R',
    'prepare_submit_host_ms',
    'prepare_ready_host_ms',
    'prepare_event_interval_ms',
    'extra_weight_bytes',
    'numeric_status',
    'warmup',
    'repeat',
    'samples',
    'library_sha256',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def contract():
    cases = list(csv.DictReader((E / 'benchmarks/data/targets_int8_nt.csv').open()))
    assert len(cases) == 1178
    assert len({int(t['ID']) for t in cases}) == 1178
    assert sum(t['phase'] == 'decode' for t in cases) == 872
    assert sum(t['phase'] == 'prefill' for t in cases) == 306
    for t in cases:
        g, m, n, k = [int(t[x]) for x in ['G', 'M', 'N', 'K']]
        assert min(g, m, n, k) > 0
        assert g <= 65535 and m <= 1048560 and n <= 1048576
        assert 256 <= k <= 131056 and k % 16 == 0 and n % 4 == 0
        assert max(n * k, m * k, m * n) <= 2**31 - 1
        # Exact FP32 integer reference for inputs [-8,8], with TF32 disabled.
        assert k * 64 < 2**24
        work = 2 * g * m * n * k
        assert math.isclose(
            work / float(t['target_ms']) / 1e9, float(t['target_TOPS']), rel_tol=1e-10
        )
    return cases


def snapshot():
    identity = json.loads((BUILD / 'sources.json').read_text())
    for name, value in {
        **identity['extension'],
        **identity['existing_public_context_unchanged'],
    }.items():
        assert digest(ROOT / name) == value, name
    return dict(
        api_library=digest(BUILD / 'libck_tile_int8_nt_cached.so'),
        benchmark_library=digest(BUILD / 'libck_tile_int8_nt_cached_benchmark.so'),
        target_csv=digest(E / 'benchmarks/data/targets_int8_nt.csv'),
        runner=digest(Path(__file__)),
        compiled_sources=identity,
        test_sources={
            str(p.relative_to(E)): digest(p)
            for p in (E / 'tests').glob('*.py')
        },
    )


def export(out, cases, records, sha):
    by_id = {x['ID']: x for x in records}
    temporary = out / 'results.csv.tmp'
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for t in cases:
            row = {x: t[x] for x in ['ID', 'phase', 'M', 'N', 'K', 'G']}
            row.update(
                dtype='INT8xINT8->INT32_no_scale',
                layout='NT',
                BLAS_target_ms=t['target_ms'],
                BLAS_target_TOPS=t['target_TOPS'],
                numeric_status='PENDING',
                library_sha256=sha,
            )
            if int(t['ID']) in by_id:
                row.update(
                    {k: v for k, v in by_id[int(t['ID'])].items() if k in FIELDS}
                )
            writer.writerow(row)
    temporary.replace(out / 'results.csv')


def verify_prefix(cases, records, sha):
    assert len(records) <= len(cases)
    for t, rec in zip(cases, records):
        assert all(rec[k] == int(t[k]) for k in ['ID', 'M', 'N', 'K', 'G'])
        assert rec['phase'] == t['phase']
        assert rec['dtype'] == 'INT8xINT8->INT32_no_scale' and rec['layout'] == 'NT'
        assert rec['BLAS_target_ms'] == float(t['target_ms'])
        assert rec['BLAS_target_TOPS'] == float(t['target_TOPS'])
        assert rec['library_sha256'] == sha and rec['numeric_status'] == 'PASS'
        assert (
            rec['bad_elements'] == 0 and rec['guard'] and rec['input_samples_unchanged']
        )
        assert (rec['warmup'], rec['repeat'], rec['samples']) == (10, 50, 3)
        assert len(rec['samples_ms']) == 3 and min(rec['samples_ms']) > 0
        assert rec['CK_ms'] == statistics.median(rec['samples_ms'])
        work = 2 * rec['G'] * rec['M'] * rec['N'] * rec['K']
        assert math.isclose(work / rec['CK_ms'] / 1e9, rec['CK_TOPS'], rel_tol=1e-12)
        assert math.isclose(
            rec['CK_TOPS'] / float(t['target_TOPS']), rec['R'], rel_tol=1e-12
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not args.plan_only:
        parser.error("Use --plan-only or run benchmarks/run_scaled.py for BF16/FP16.")
    cases = contract()
    frozen = snapshot()
    print(
        json.dumps(
            dict(
                status='CPU_CONTRACT_PASS_GPU_NOT_RUN',
                shapes=len(cases),
                prefill=306,
                decode=872,
                snapshot=frozen,
            ),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
