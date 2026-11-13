"""One-round scaled INT8 NT1191-shape benchmark,BF16 and FP16 on physical GPU7."""

from pathlib import Path
import argparse, csv, json, os, sys, time, datetime, hashlib, statistics, math, resource, subprocess, traceback, gc

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
E = Path(__file__).resolve().parents[1]
B = Path(os.environ['CK_INT8_CACHED_BUILD'])
sys.path.insert(0, str(E / 'tests'))
import validate_all_1178 as original

OUTPUTS = ('BF16', 'FP16')
PATH_FIELDS = ('pipeline', 'route', 'legacy_instance', 'legacy_fallback_instance')

FIELDS = [
    'ID',
    'phase',
    'M',
    'N',
    'K',
    'E',
    'G',
    'dtype',
    'layout',
    'activation_scale',
    'weight_scale',
    'BF16_ms',
    'BF16_TOPS',
    'BF16_status',
    'FP16_ms',
    'FP16_TOPS',
    'FP16_status',
    *(f'{kind}_{field}' for kind in OUTPUTS for field in PATH_FIELDS),
    'prepare_submit_host_ms',
    'prepare_ready_host_ms',
    'prepare_event_interval_ms',
    'extra_weight_bytes',
    'warmup',
    'repeat',
    'samples',
    'library_sha256',
    'BLAS_target_ms',
    'BLAS_target_TOPS',
    'BF16_R',
    'FP16_R',
]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export(out, cases, records, sha):
    byid = {r['ID']: r for r in records}
    tmp = out / 'results.csv.tmp'
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for case in cases:
            row = {
                **case,
                'dtype': 'INT8xINT8',
                'layout': 'NT',
                'activation_scale': 'FP32_per_group_scalar',
                'weight_scale': 'FP32_per_channel_N',
                'BLAS_target_ms': case['target_ms'],
                'BLAS_target_TOPS': case['target_TOPS'],
                'BF16_status': 'PENDING',
                'FP16_status': 'PENDING',
                'library_sha256': sha,
            }
            row.update(byid.get(case['ID'], {}))
            w.writerow(row)
    tmp.replace(out / 'results.csv')
    for kind in OUTPUTS:
        fields = [
            'ID',
            'phase',
            'M',
            'N',
            'K',
            'E',
            'G',
            'dtype',
            'layout',
            'activation_scale',
            'weight_scale',
            'CK_ms',
            'CK_TOPS',
            'numeric_status',
            *PATH_FIELDS,
            'warmup',
            'repeat',
            'samples',
            'library_sha256',
            'BLAS_target_ms',
            'BLAS_target_TOPS',
            'R',
        ]
        tmp = out / (kind.lower() + '.csv.tmp')
        with tmp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for case in cases:
                rec = byid.get(case['ID'], {})
                row = {k: rec.get(k, case.get(k, '')) for k in fields}
                row.update(
                    dtype='INT8xINT8->' + kind,
                    layout='NT',
                    activation_scale='FP32_per_group_scalar',
                    weight_scale='FP32_per_channel_N',
                    CK_ms=rec.get(kind + '_ms', ''),
                    CK_TOPS=rec.get(kind + '_TOPS', ''),
                    numeric_status=rec.get(kind + '_status', 'PENDING'),
                    **{field: rec.get(f'{kind}_{field}', '') for field in PATH_FIELDS},
                    library_sha256=sha,
                    BLAS_target_ms=case['target_ms'],
                    BLAS_target_TOPS=case['target_TOPS'],
                    R=rec.get(kind + '_R', ''),
                )
                w.writerow(row)
        tmp.replace(out / (kind.lower() + '.csv'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    source = list(
        csv.DictReader((E / 'benchmarks/data/targets_max_all_1191.csv').open())
    )
    assert len(source) == 1191 and [int(x['ID']) for x in source] == list(
        range(1, 1192)
    )
    cases = [
        dict(
            ID=int(x['ID']),
            phase=x['phase'],
            M=int(x['M']),
            N=int(x['N']),
            K=int(x['K']),
            E=int(x['G']),
            G=int(x['G']),
            target_ms=float(x['target_ms']),
            target_TOPS=float(x['target_TOPS']),
        )
        for x in source
    ]
    assert (
        sum(x['phase'] == 'decode' for x in cases) == 881
        and sum(x['phase'] == 'prefill' for x in cases) == 310
    )
    for case in cases:
        assert (
            min(case[x] for x in ['M', 'N', 'K', 'G', 'target_TOPS', 'target_ms']) > 0
        )
        assert math.isclose(
            2 * case['G'] * case['M'] * case['N'] * case['K'] / case['target_ms'] / 1e9,
            case['target_TOPS'],
            rel_tol=1e-10,
        )
    if args.limit:
        cases = cases[: args.limit]
    frozen = original.snapshot()
    assert not json.loads((B / 'exported_api_symbols.json').read_text())[
        'experimental_tuning'
    ]
    version = os.environ.get('CK_INT8_TEST_VERSION', 'v136')
    assert version.startswith('v') and version[1:].isdigit(), 'Invalid gate version'
    validation = json.loads(
        (E / f'results/max_all_{version}_gates/scaled_validation.json').read_text()
    )
    assert (
        validation['status'] == 'PASS'
        and validation['library_sha256'] == frozen['api_library']
    )
    for name in [
        f'cached_api_{version}_boundaries.json',
        f'cached_api_{version}_lifetime.json',
    ]:
        check = json.loads((E / 'results' / name).read_text())
        assert (
            check['status'] == 'PASS'
            and check['library_sha256'] == frozen['api_library']
        )
    assert all(
        os.environ.get(x) == '7'
        for x in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'DTK_VISIBLE_DEVICES']
    )
    import torch
    from bindings import C, Weight, init, lib, ok, weight_info, plan_info
    from test_scaled_api import Group2

    lib.cached_i8_bench_pipeline.argtypes = [C.c_uint, C.c_uint]
    lib.cached_i8_bench_pipeline.restype = C.c_char_p

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.startswith('gfx936')
    lib.hipDeviceGetPCIBusId.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.hipDeviceGetPCIBusId.restype = C.c_int
    bus = C.create_string_buffer(64)
    assert lib.hipDeviceGetPCIBusId(bus, 64, 0) == 0
    pci = Path('/sys/class/drm/renderD135/device').resolve().name.lower()
    assert bus.value.decode().lower() == pci
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    recordpath = out / 'records.jsonl'
    assert (
        recordpath.exists() == args.resume
    ), 'Explicit --resume required for existing records'
    identity = dict(
        snapshot=frozen,
        runner_sha256=digest(Path(__file__)),
        cases_sha256=digest(E / 'benchmarks/data/targets_max_all_1191.csv'),
    )
    records = (
        [json.loads(x) for x in recordpath.read_text().splitlines()]
        if args.resume
        else []
    )
    if args.resume:
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['identity'] == identity
        for case, rec in zip(cases, records):
            assert all(
                case[k] == rec[k] for k in ['ID', 'M', 'N', 'K', 'G', 'E', 'phase']
            )
            assert rec['library_sha256'] == frozen['api_library']
            assert rec['BF16_status'] == rec['FP16_status'] == 'PASS'
            for kind in ['BF16', 'FP16']:
                assert all(f'{kind}_{field}' in rec for field in PATH_FIELDS)
                assert rec[kind + '_ms'] == statistics.median(rec[kind + '_samples_ms'])
        original.atomic(
            out / ('resume_' + str(time.time_ns()) + '.json'),
            dict(prefix_records=len(records), identity=identity),
        )
    else:
        manifest = dict(
            identity=identity,
            source_revision=subprocess.check_output(
                ['git', '-C', str(E.parents[2]), 'rev-parse', 'HEAD'], text=True
            ).strip(),
            gpu=7,
            pci=pci,
            device=str(prop),
            total_shapes=len(cases),
            outputs=['BF16', 'FP16'],
            formal_rounds=1,
            warmup=10,
            repeat=50,
            samples=3,
            statistic='median',
            input_range=[-8, 8],
            scaleA='FP32 scalar per group,random[0.002,0.01)',
            scaleB='FP32 vector[N] per group,random[0.002,0.01)',
            validation='Full output:FP32 exact integer dot for bounded inputs,TF32 off;FP32(scaleA*scaleB);FP32 multiply;selected16bit RNE;guards;sampled input/scale immutability',
            timing='C++ HIP events around50 GEMM runs;10 warmups per sample;3 samples;prepare excluded',
            preparation='One static-weight prepare per shape reused for both output types;host/event preparation measurements are separate',
            order='BF16/FP16 order alternates by ID',
            blas='Frozen workbook int8_NT columnP Max ALL engineering target;maximum of INT32 and scaled BLAS modes,not a claim of identical per-channel BLAS semantics',
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        original.atomic(out / 'manifest.json', manifest)
    export(out, cases, records, frozen['api_library'])

    def active(case, stage, **more):
        original.atomic(
            out / 'active.json',
            dict(**case, stage=stage, completed=len(records), total=len(cases), **more),
        )

    try:
        with recordpath.open('a' if args.resume else 'w') as f:
            for case in cases[len(records) :]:
                idx, g, m, n, k = [case[x] for x in ['ID', 'G', 'M', 'N', 'K']]
                active(case, 'allocate')
                packed = n % 256 == 0 and k % 128 == 0 and 512 <= k < 16384
                span = m * n
                capacity = (span + 7) // 8 * 8
                estimate = (
                    g
                    * (
                        m * k
                        + n * k * (1 + int(packed))
                        + 4 * (capacity + 32)
                        + 4 * (n + 1)
                    )
                    + 4 * n * k
                    + 16 * 128 * (k + n)
                    + 1024**3
                )
                free, _ = torch.cuda.mem_get_info()
                assert estimate < free, (
                    'Memory budget;keep original groups',
                    idx,
                    estimate,
                    free,
                )
                torch.manual_seed(20261104 + idx)
                a = torch.randint(-8, 9, (g, m, k), device='cuda', dtype=torch.int8)
                b = torch.randint(-8, 9, (g, n, k), device='cuda', dtype=torch.int8)
                sa = (
                    torch.rand((g,), device='cuda', dtype=torch.float32) * 0.008 + 0.002
                )
                sb = (
                    torch.rand((g, n), device='cuda', dtype=torch.float32) * 0.008
                    + 0.002
                )
                raw = {
                    kind: torch.full(
                        (g, capacity + 32), 117, device='cuda', dtype=dtype
                    )
                    for kind, dtype in [
                        ('BF16', torch.bfloat16),
                        ('FP16', torch.float16),
                    ]
                }
                output = {
                    kind: v[:, 16 : 16 + span].view(g, m, n) for kind, v in raw.items()
                }
                sampled = []
                for v in [a, b, sa, sb]:
                    inds = (
                        torch.arange(min(256, v.numel()), device='cuda')
                        * (v.numel() - 1)
                        // max(min(256, v.numel()) - 1, 1)
                    )
                    sampled.append((inds, v.reshape(-1)[inds].clone()))
                wd = (Weight * g)(
                    *[
                        init(Weight, data=b[i].data_ptr(), n=n, k=k, row_stride=k)
                        for i in range(g)
                    ]
                )
                handle = C.c_void_p()
                plans = {}
                benches = {}
                try:
                    active(case, 'prepare')
                    load = torch.cuda.Stream()
                    load.wait_stream(torch.cuda.current_stream())
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin.record(load)
                    start = time.perf_counter()
                    ok(
                        lib.ck_tile_int8_nt_cached_weights_create_v1(
                            wd, g, load.cuda_stream, C.byref(handle)
                        )
                    )
                    submit = (time.perf_counter() - start) * 1000
                    end.record(load)
                    end.synchronize()
                    ready = (time.perf_counter() - start) * 1000
                    wi = weight_info(handle)
                    rec = dict(
                        **case,
                        dtype='INT8xINT8',
                        layout='NT',
                        BLAS_target_ms=case['target_ms'],
                        BLAS_target_TOPS=case['target_TOPS'],
                        activation_scale='FP32_per_group_scalar',
                        weight_scale='FP32_per_channel_N',
                        prepare_submit_host_ms=submit,
                        prepare_ready_host_ms=ready,
                        prepare_event_interval_ms=begin.elapsed_time(end),
                        extra_weight_bytes=wi.packed_bytes,
                        warmup=10,
                        repeat=50,
                        samples=3,
                        library_sha256=frozen['api_library'],
                    )
                    for code, kind in [(1, 'BF16'), (2, 'FP16')]:
                        gd = (Group2 * g)()
                        for i in range(g):
                            d = gd[i]
                            d.struct_size = C.sizeof(d)
                            d.abi_version = 2
                            d.a = a[i].data_ptr()
                            d.c = output[kind][i].data_ptr()
                            d.m = m
                            d.a_row_stride = k
                            d.c_row_stride = n
                            d.output_type = code
                            d.scale_a = sa[i:].data_ptr()
                            d.scale_b = sb[i].data_ptr()
                            d.scale_a_mode = 0
                            d.scale_b_mode = 1
                        plan = C.c_void_p()
                        ok(
                            lib.ck_tile_int8_nt_cached_plan_create_v2(
                                handle, gd, g, C.byref(plan)
                            )
                        )
                        plans[kind] = plan
                        pi = plan_info(plan)
                        assert (pi.group_count, pi.max_m, pi.n, pi.k) == (g, m, n, k)
                        pipeline = lib.cached_i8_bench_pipeline(pi.route, pi.legacy_instance)
                        assert pipeline, (idx, kind, pi.route, pi.legacy_instance)
                        rec.update({
                            f'{kind}_route': pi.route,
                            f'{kind}_pipeline': pipeline.decode('ascii'),
                            f'{kind}_legacy_instance': pi.legacy_instance if pi.route == 0 else '',
                            f'{kind}_legacy_fallback_instance': pi.legacy_instance if pi.route else '',
                        })
                        benches[kind] = lib.cached_i8_bench_create(plan)
                        assert benches[kind], lib.cached_i8_bench_error().decode()
                    torch.cuda.synchronize()
                    for kind in (['BF16', 'FP16'] if idx % 2 else ['FP16', 'BF16']):
                        active(case, 'timing', output_type=kind, route=rec[f'{kind}_route'])
                        ts = [
                            float(lib.cached_i8_bench_run(benches[kind], 10, 50, None))
                            for _ in range(3)
                        ]
                        assert min(ts) > 0 and all(
                            math.isfinite(t) for t in ts
                        ), lib.cached_i8_bench_error().decode()
                        ms = statistics.median(ts)
                        rec.update(
                            {
                                kind + '_samples_ms': ts,
                                kind + '_ms': ms,
                                kind + '_TOPS': 2 * g * m * n * k / ms / 1e9,
                                kind
                                + '_R': 2
                                * g
                                * m
                                * n
                                * k
                                / ms
                                / 1e9
                                / case['target_TOPS'],
                            }
                        )
                    active(
                        case,
                        'reference',
                        BF16_ms=rec['BF16_ms'],
                        FP16_ms=rec['FP16_ms'],
                    )
                    bad = {'BF16': 0, 'FP16': 0}
                    for group in range(g):
                        bf = b[group].float()
                        factor = sa[group] * sb[group]
                        for row in range(0, m, 128):
                            dot = a[group, row : row + 128].float() @ bf.T
                            scaled = dot * factor[None, :]
                            for kind, dtype in [
                                ('BF16', torch.bfloat16),
                                ('FP16', torch.float16),
                            ]:
                                ref = scaled.to(dtype)
                                bad[kind] += int(
                                    (output[kind][group, row : row + 128] != ref).sum()
                                )
                            del dot, scaled, ref
                        del bf, factor
                    guards = {
                        kind: bool(
                            (v[:, :16] == 117).all() & (v[:, 16 + span :] == 117).all()
                        )
                        for kind, v in raw.items()
                    }
                    unchanged = all(
                        torch.equal(v.reshape(-1)[inds], old)
                        for v, (inds, old) in zip([a, b, sa, sb], sampled)
                    )
                    assert (
                        all(x == 0 for x in bad.values())
                        and all(guards.values())
                        and unchanged
                    ), ('NUMERIC_FAIL', idx, bad, guards, unchanged)
                    rec.update(
                        BF16_status='PASS',
                        FP16_status='PASS',
                        bad_elements=bad,
                        guards=guards,
                        input_scale_samples_unchanged=unchanged,
                    )
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    records.append(rec)
                    export(out, cases, records, frozen['api_library'])
                    original.atomic(
                        out / 'progress.json',
                        dict(
                            completed=len(records),
                            total=len(cases),
                            BF16_PASS=len(records),
                            FP16_PASS=len(records),
                            last_ID=idx,
                            updated_utc=datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                        ),
                    )
                    print(
                        json.dumps(
                            {
                                key: rec[key]
                                for key in [
                                    'ID',
                                    'M',
                                    'N',
                                    'K',
                                    'G',
                                    'BF16_ms',
                                    'BF16_TOPS',
                                    'FP16_ms',
                                    'FP16_TOPS',
                                    'BF16_status',
                                    'FP16_status',
                                    'BF16_route',
                                    'FP16_route',
                                ]
                            }
                        ),
                        flush=True,
                    )
                finally:
                    for bench in benches.values():
                        lib.cached_i8_bench_destroy(bench)
                    for plan in plans.values():
                        ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
                    if handle.value:
                        ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(handle))
                del a, b, sa, sb, raw, output, sampled, wd, gd
                torch.cuda.empty_cache()
        assert len(records) == len(cases)
        assert (
            original.snapshot() == frozen
        ), 'Compiled sources changed during full acceptance'
        original.atomic(
            out / 'done.json',
            dict(
                status='PASS',
                shapes=len(cases),
                BF16_PASS=len(records),
                FP16_PASS=len(records),
                library_sha256=frozen['api_library'],
                strict_at_least_target={
                    kind: sum(r[kind + '_R'] >= 1 for r in records)
                    for kind in ['BF16', 'FP16']
                },
                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        print('DONE', len(records), 'shapes;both outputs PASS', flush=True)
    except BaseException:
        original.atomic(
            out / 'failure.json',
            dict(
                completed=len(records),
                traceback=traceback.format_exc(),
                time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == '__main__':
    main()
