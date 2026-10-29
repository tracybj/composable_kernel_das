"""Frozen-library acceptance through the additive INT8 cached-weight API.

All original shapes/groups; preparation and storage are reported separately.
--plan-only performs CPU contract checks and never imports Torch or loads HIP.
"""
from pathlib import Path
import argparse
import csv
import datetime
import gc
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import time
import traceback

E = Path(__file__).resolve().parents[2]
ROOT = E.parents[2]
BUILD = E / 'build/cached_v97'
FIELDS = ['ID', 'phase', 'M', 'N', 'K', 'G', 'dtype', 'layout', 'route',
          'legacy_instance', 'CK_ms', 'CK_TOPS', 'BLAS_target_ms',
          'BLAS_target_TOPS', 'R', 'prepare_submit_host_ms',
          'prepare_ready_host_ms', 'prepare_event_interval_ms',
          'extra_weight_bytes', 'numeric_status', 'warmup', 'repeat',
          'samples', 'library_sha256']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def contract():
    cases = list(csv.DictReader((E / 'docs/targets_int8_nt.csv').open()))
    assert len(cases) == 1178
    assert len({int(t['ID']) for t in cases}) == 1178
    assert sum(t['phase'] == 'decode' for t in cases) == 872
    assert sum(t['phase'] == 'prefill' for t in cases) == 306
    for t in cases:
        g, m, n, k = [int(t[x]) for x in ['G', 'M', 'N', 'K']]
        assert min(g, m, n, k) > 0
        assert g <= 65535 and m <= 1048560 and n <= 1048576
        assert 256 <= k <= 131056 and k % 16 == 0 and n % 4 == 0
        assert max(n*k, m*k, m*n) <= 2**31 - 1
        # Exact FP32 integer reference for inputs [-8,8], with TF32 disabled.
        assert k * 64 < 2**24
        work = 2 * g * m * n * k
        assert math.isclose(work / float(t['target_ms']) / 1e9,
                            float(t['target_TOPS']), rel_tol=1e-10)
    return cases


def snapshot():
    identity = json.loads((BUILD / 'sources.json').read_text())
    for name, value in {**identity['extension'],
                        **identity['existing_public_context_unchanged']}.items():
        assert digest(ROOT / name) == value, name
    return dict(api_library=digest(BUILD / 'libck_tile_int8_nt_cached.so'),
                benchmark_library=digest(BUILD / 'libck_tile_int8_nt_cached_benchmark.so'),
                target_csv=digest(E / 'docs/targets_int8_nt.csv'),
                runner=digest(Path(__file__)),
                compiled_sources=identity,
                test_sources={str(p.relative_to(E)): digest(p)
                              for p in (E / 'cached_weights/tests').glob('*.py')})


def export(out, cases, records, sha):
    by_id = {x['ID']: x for x in records}
    temporary = out / 'results.csv.tmp'
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for t in cases:
            row = {x: t[x] for x in ['ID', 'phase', 'M', 'N', 'K', 'G']}
            row.update(dtype='INT8xINT8->INT32_no_scale', layout='NT',
                       BLAS_target_ms=t['target_ms'],
                       BLAS_target_TOPS=t['target_TOPS'],
                       numeric_status='PENDING', library_sha256=sha)
            if int(t['ID']) in by_id:
                row.update({k: v for k, v in by_id[int(t['ID'])].items() if k in FIELDS})
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
        assert rec['bad_elements'] == 0 and rec['guard'] and rec['input_samples_unchanged']
        assert (rec['warmup'], rec['repeat'], rec['samples']) == (10, 50, 3)
        assert len(rec['samples_ms']) == 3 and min(rec['samples_ms']) > 0
        assert rec['CK_ms'] == statistics.median(rec['samples_ms'])
        work = 2 * rec['G'] * rec['M'] * rec['N'] * rec['K']
        assert math.isclose(work / rec['CK_ms'] / 1e9, rec['CK_TOPS'], rel_tol=1e-12)
        assert math.isclose(rec['CK_TOPS'] / float(t['target_TOPS']), rec['R'], rel_tol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    cases = contract()
    frozen = snapshot()
    if args.plan_only:
        print(json.dumps(dict(status='CPU_CONTRACT_PASS_GPU_NOT_RUN', shapes=len(cases),
                              prefill=306, decode=872, snapshot=frozen), indent=2))
        return
    assert args.output, '--output is required'
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '7'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '7'
    assert Path(os.environ.get('CK_INT8_CACHED_BENCH_LIB',
                str(BUILD / 'libck_tile_int8_nt_cached_benchmark.so'))).resolve() == (
                BUILD / 'libck_tile_int8_nt_cached_benchmark.so').resolve()
    for name in ['cached_api_v97_boundaries.json', 'cached_api_v97_lifetime.json']:
        check = json.loads((E / 'results' / name).read_text())
        assert check['status'] == 'PASS' and check['library_sha256'] == frozen['api_library']
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch
    from bindings import C, Group, Weight, init, lib, ok, plan_info, weight_info
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), 'GPU unavailable: do not create/overwrite results'
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx936')
    lib.hipDeviceGetPCIBusId.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.hipDeviceGetPCIBusId.restype = C.c_int
    bus = C.create_string_buffer(64)
    assert lib.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
    physical7_pci = Path('/sys/class/drm/renderD135/device').resolve().name.lower()
    assert bus.value.decode().lower() == physical7_pci, 'Logical device0 must be physical7'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records_path = out / 'records.jsonl'
    assert records_path.exists() == args.resume, 'Inspect prior results; use explicit --resume'
    records = [json.loads(x) for x in records_path.read_text().splitlines()] if args.resume else []
    if args.resume:
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['snapshot'] == frozen
        verify_prefix(cases, records, frozen['api_library'])
        stamp = time.time_ns()
        atomic(out / f'resume_{stamp}.json', dict(prefix_records=len(records),
               prefix_sha256=digest(records_path), snapshot=frozen))
    else:
        atomic(out / 'manifest.json', dict(snapshot=frozen, total_shapes=1178,
               gpu=7, device_pci=physical7_pci, source_revision=subprocess.check_output(
                   ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
               warmup=10, repeat=50, samples=3, input_range=[-8, 8],
               validation='Full output;FP32 exact integer reference;TF32 off;128-row chunks;guards;sampled input immutability',
               preparation='One prepare per shape outside GEMM timing;event interval includes allocation/metadata/host submission gaps,not pure pack-kernel time',
               preparation_host='submit host time and host time until ready,one measured preparation,not repeated median',
               precision_isolation='Independent INT8-only API;existing FP16/BF16 ABI and defaults unchanged',
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    export(out, cases, records, frozen['api_library'])
    try:
        with records_path.open('a' if args.resume else 'w') as stream:
            for t in cases[len(records):]:
                idx, g, m, n, k = [int(t[x]) for x in ['ID', 'G', 'M', 'N', 'K']]
                active = dict(ID=idx, M=m, N=n, K=k, G=g, phase=t['phase'])
                atomic(out / 'active.json', dict(**active, stage='allocate'))
                packable = n % 256 == 0 and k % 128 == 0 and 1024 <= k < 16384
                estimate = g * (m*k + n*k*(1 + int(packable)) + 4*(m*n+32))
                estimate += 4*n*k + 16*128*(k+n) + 1024**3
                free, _ = torch.cuda.mem_get_info()
                assert estimate < free, ('Memory budget; keep original G', idx, estimate, free)
                torch.manual_seed(20261029 + idx)
                a = torch.randint(-8, 9, (g, m, k), device='cuda', dtype=torch.int8)
                b = torch.randint(-8, 9, (g, n, k), device='cuda', dtype=torch.int8)
                raw = torch.full((g, m*n+32), 117, device='cuda', dtype=torch.int32)
                c = raw[:, 16:-16].view(g, m, n)
                sampled = []
                for v in [a, b]:
                    inds = torch.arange(256, device='cuda') * (v.numel()-1) // 255
                    sampled.append((inds, v.reshape(-1)[inds].clone()))
                wd = (Weight*g)(*[init(Weight, data=b[i].data_ptr(), n=n, k=k, row_stride=k) for i in range(g)])
                gd = (Group*g)(*[init(Group, a=a[i].data_ptr(), c=c[i].data_ptr(),
                                      m=m, a_row_stride=k, c_row_stride=n) for i in range(g)])
                handle, plan, bench = C.c_void_p(), C.c_void_p(), None
                try:
                    atomic(out / 'active.json', dict(**active, stage='prepare'))
                    load = torch.cuda.Stream()
                    load.wait_stream(torch.cuda.current_stream())
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(load)
                    start = time.perf_counter()
                    ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd, g, load.cuda_stream, C.byref(handle)))
                    submit_ms = (time.perf_counter() - start) * 1e3
                    end.record(load)
                    end.synchronize()
                    ready_ms = (time.perf_counter() - start) * 1e3
                    event_ms = begin.elapsed_time(end)
                    wi = weight_info(handle)
                    assert wi.packed_bytes == (g*n*k if packable else 0)
                    ok(lib.ck_tile_int8_nt_cached_plan_create_v1(handle, gd, g, C.byref(plan)))
                    pi = plan_info(plan)
                    assert (pi.group_count, pi.max_m, pi.n, pi.k) == (g, m, n, k)
                    bench = lib.cached_i8_bench_create(plan)
                    assert bench, lib.cached_i8_bench_error().decode()
                    c.fill_(123456789)
                    torch.cuda.synchronize()
                    atomic(out / 'active.json', dict(**active, stage='timing', route=pi.route))
                    ts = [float(lib.cached_i8_bench_run(bench, 10, 50, None)) for _ in range(3)]
                    assert min(ts) > 0 and all(math.isfinite(x) for x in ts)
                    atomic(out / 'active.json', dict(**active, stage='reference'))
                    bad = 0
                    for group in range(g):
                        bf = b[group].float()
                        for row in range(0, m, 128):
                            ref = (a[group, row:row+128].float() @ bf.T).to(torch.int32)
                            bad += int((c[group, row:row+128] != ref).sum())
                        del bf, ref
                    guard = bool((raw[:, :16] == 117).all() & (raw[:, -16:] == 117).all())
                    unchanged = all(torch.equal(v.reshape(-1)[inds], old)
                                    for v, (inds, old) in zip([a, b], sampled))
                    assert bad == 0 and guard and unchanged, (idx, bad, guard, unchanged)
                    ms = statistics.median(ts)
                    tops = 2*g*m*n*k / ms / 1e9
                    rec = dict(**active, dtype='INT8xINT8->INT32_no_scale', layout='NT',
                               route=pi.route, legacy_instance=pi.legacy_instance,
                               CK_ms=ms, CK_TOPS=tops, samples_ms=ts,
                               BLAS_target_ms=float(t['target_ms']), BLAS_target_TOPS=float(t['target_TOPS']),
                               R=tops / float(t['target_TOPS']), prepare_submit_host_ms=submit_ms,
                               prepare_ready_host_ms=ready_ms, prepare_event_interval_ms=event_ms,
                               extra_weight_bytes=wi.packed_bytes, numeric_status='PASS',
                               bad_elements=bad, guard=guard, input_samples_unchanged=unchanged,
                               warmup=10, repeat=50, samples=3, library_sha256=frozen['api_library'])
                    stream.write(json.dumps(rec) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    records.append(rec)
                    export(out, cases, records, frozen['api_library'])
                    progress = dict(completed=len(records), total=1178, last_ID=idx,
                                    ge1=sum(x['R'] >= 1 for x in records))
                    atomic(out / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
                finally:
                    if bench:
                        lib.cached_i8_bench_destroy(bench)
                    if plan.value:
                        ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                    if handle.value:
                        ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(handle))
                del a, b, raw, c, wd, gd, sampled, v, inds
                gc.collect()
                torch.cuda.empty_cache()
        assert snapshot() == frozen
        verify_prefix(cases, records, frozen['api_library'])
        assert len(records) == 1178
        atomic(out / 'done.json', dict(status='PASS', numeric_pass=1178,
               ge1=sum(x['R'] >= 1 for x in records), min_R=min(x['R'] for x in records),
               median_R=statistics.median(x['R'] for x in records), snapshot=frozen))
    except BaseException:
        atomic(out / f'error_{time.time_ns()}.json', dict(traceback=traceback.format_exc(),
               completed=len(records), snapshot=frozen))
        raise


if __name__ == '__main__':
    main()
