"""Matched small-M data paths. Profile one full grouped launch, never one expert.

Use xprof's GroupedInt8NtScaledTileKernel substring filter; setup/reference kernels
are deliberately outside that filter. Baseline timings are collected separately.
"""
from pathlib import Path
import argparse, ctypes as C, hashlib, json, os, statistics, sys

E = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(E / 'cached_weights/tests'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', choices=['all', 'both_lds', 'a_reg', 'raw_reg', 'packed_reg'], default='all')
    ap.add_argument('--mode', choices=['baseline', 'profile'], default='baseline')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert all(os.environ.get(k) == '7' for k in ['HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'DTK_VISIBLE_DEVICES'])
    assert args.mode != 'profile' or args.candidate != 'all'
    import torch
    from bindings import Weight, init, lib, ok, plan_info
    from test_scaled_api import Group2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261106)
    m, n, k, g = 1, 4096, 6144, 8
    a = torch.randint(-128, 128, (g, m, k), device='cuda', dtype=torch.int8)
    b = torch.randint(-128, 128, (g, n, k), device='cuda', dtype=torch.int8)
    sa = torch.rand(g, device='cuda') * .008 + .002
    sb = torch.rand(g, n, device='cuda') * .008 + .002
    c = torch.full((g, m, n + 8), 77, device='cuda', dtype=torch.bfloat16)
    refs = [((a[i].double() @ b[i].double().T).to(torch.int32).float() *
             (sa[i] * sb[i])[None, :]).to(torch.bfloat16) for i in range(g)]
    hashes = {name: hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
              for name, x in [('a', a), ('b', b), ('sa', sa), ('sb', sb)]}
    wd = (Weight * g)(*[init(Weight, data=b[i].data_ptr(), n=n, k=k, row_stride=k) for i in range(g)])
    weights = C.c_void_p()
    ok(lib.ck_tile_int8_nt_cached_weights_create_v1(wd, g, None, C.byref(weights)))
    candidates = {'both_lds': (0, 8734), 'a_reg': (0, 9041),
                  'raw_reg': (0, 9054), 'packed_reg': (32, None)}
    plans, benches, samples = {}, {}, {}
    records = []
    try:
        for name, (route, legacy) in candidates.items():
            if args.candidate != 'all' and name != args.candidate:
                continue
            os.environ['CK_INT8_TUNE_ROUTE'] = str(route)
            os.environ.pop('CK_INT8_TUNE_LEGACY', None)
            if legacy is not None:
                os.environ['CK_INT8_TUNE_LEGACY'] = str(legacy)
            gd = (Group2 * g)()
            for i in range(g):
                d = gd[i]
                d.struct_size, d.abi_version = C.sizeof(d), 2
                d.a, d.c, d.m = a[i].data_ptr(), c[i].data_ptr(), m
                d.a_row_stride, d.c_row_stride, d.output_type = k, n + 8, 1
                d.scale_a, d.scale_b, d.scale_b_mode = sa[i:].data_ptr(), sb[i].data_ptr(), 1
            plan = C.c_void_p()
            ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights, gd, g, C.byref(plan)))
            pi = plan_info(plan)
            assert pi.route == route and (legacy is None or pi.legacy_instance == legacy)
            plans[name] = plan
            if args.mode == 'baseline':
                bench = lib.cached_i8_bench_create(plan)
                assert bench
                benches[name], samples[name] = bench, []
        torch.cuda.synchronize()
        if args.mode == 'baseline':
            keys = list(plans)
            for sample in range(3):
                order = keys[sample % len(keys):] + keys[:sample % len(keys)]
                if sample % 2:
                    order = order[::-1]
                for name in order:
                    ms = float(lib.cached_i8_bench_run(benches[name], 10, 50, None))
                    assert ms > 0, lib.cached_i8_bench_error().decode()
                    samples[name].append(ms)
        # Exactly one selected grouped launch in profile mode. Validation remains
        # mandatory after replay; no host-side expert loop dispatches CK kernels.
        for name, plan in plans.items():
            c.fill_(77)
            torch.cuda.synchronize()
            ok(lib.ck_tile_int8_nt_cached_run_v1(plan, None))
            torch.cuda.synchronize()
            bad = sum(int((c[i, :, :n].view(torch.int16) != refs[i].view(torch.int16)).sum()) for i in range(g))
            assert bad == 0 and bool((c[:, :, n:] == 77).all()), (name, bad)
            row = dict(candidate=name, M=m, N=n, K=k, G=g, status='PASS',
                       bad_elements=bad, scope='one complete grouped kernel, all G=8',
                       samples_ms=samples.get(name))
            if name in samples:
                row['ms'] = statistics.median(samples[name])
                row['TOPS'] = 2 * g * m * n * k / row['ms'] / 1e9
            records.append(row)
    finally:
        for bench in benches.values():
            lib.cached_i8_bench_destroy(bench)
        for plan in plans.values():
            ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))
        ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))
    build = Path(os.environ['CK_INT8_CACHED_BUILD'])
    report = dict(mode=args.mode, input_sha256=hashes, output='BF16', scale='scalar A, per-channel B',
                  library=json.loads((build / 'exported_api_symbols.json').read_text()), records=records,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
