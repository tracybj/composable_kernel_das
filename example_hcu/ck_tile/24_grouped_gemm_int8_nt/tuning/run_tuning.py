"""Offline tuning of compiled scaled INT8 NT instances."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

EXAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE / "tests"))
from validate_all_1178 import snapshot

import torch
from bindings import C, Weight, init, lib, ok, plan_info, weight_info
from test_scaled_api import Group2


class Candidate(C.Structure):
    _fields_ = [
        ("route", C.c_uint32),
        ("legacy_instance", C.c_uint32),
        ("bm", C.c_int32),
        ("bn", C.c_int32),
        ("bk", C.c_int32),
        ("threads", C.c_int32),
        ("packed_weight", C.c_uint32),
    ]

    def key(self):
        return self.route, self.legacy_instance

    def fields(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


lib.ck_tile_int8_nt_tuning_list_v1.argtypes = [
    C.c_void_p, C.POINTER(Group2), C.c_int32, C.POINTER(Candidate),
    C.c_size_t, C.POINTER(C.c_size_t),
]
lib.ck_tile_int8_nt_tuning_plan_create_v1.argtypes = [
    C.c_void_p, C.POINTER(Group2), C.c_int32, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p),
]
DEFAULT = "default"
OUTPUTS = ((1, "BF16", torch.bfloat16), (2, "FP16", torch.float16))


def atomic(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def candidate_list(weights, descriptors):
    count = C.c_size_t()
    g = len(descriptors)
    ok(lib.ck_tile_int8_nt_tuning_list_v1(weights, descriptors, g, None, 0, C.byref(count)))
    candidates = (Candidate * count.value)()
    ok(lib.ck_tile_int8_nt_tuning_list_v1(
        weights, descriptors, g, candidates, len(candidates), C.byref(count)))
    assert count.value == len(candidates)
    result = {candidate.key(): candidate.fields() for candidate in candidates}
    assert len(result) == len(candidates) > 0
    return result


def selection(plan):
    info = plan_info(plan)
    return info.route, info.legacy_instance if info.route == 0 else 0


def make_plan(weights, descriptors, key):
    plan = C.c_void_p()
    if key == DEFAULT:
        ok(lib.ck_tile_int8_nt_cached_plan_create_v2(weights, descriptors, len(descriptors), C.byref(plan)))
    else:
        ok(lib.ck_tile_int8_nt_tuning_plan_create_v1(
            weights, descriptors, len(descriptors), *key, C.byref(plan)))
        assert selection(plan) == key
    return plan


def build_descriptors(a, c, sa, sb, output):
    g, m, k = a.shape
    n = c.shape[-1]
    descriptors = (Group2 * g)()
    for i, d in enumerate(descriptors):
        d.struct_size, d.abi_version = C.sizeof(d), 2
        d.a, d.c = a[i].data_ptr(), c[i].data_ptr()
        d.m, d.a_row_stride, d.c_row_stride = m, k, n
        d.output_type = output
        d.scale_a, d.scale_b = sa[i:].data_ptr(), sb[i].data_ptr()
        d.scale_a_mode, d.scale_b_mode = 0, 1
    return descriptors


def measure(bench, key, stream, warm, repeats):
    ms = float(lib.cached_i8_bench_run(bench, warm, repeats, stream))
    if not math.isfinite(ms) or ms <= 0:
        raise RuntimeError((key, lib.cached_i8_bench_error().decode()))
    return ms


def bench_samples(benches, keys, stream, warm, repeats, samples, seed):
    times = {key: [] for key in keys}
    for i in range(samples):
        for key in (keys if (seed + i) % 2 == 0 else reversed(keys)):
            times[key].append(measure(benches[key], key, stream, warm, repeats))
    return times


def screen_samples(benches, keys, stream, repeats, reverse):
    order = list(reversed(keys)) if reverse else keys
    first = {key: measure(benches[key], key, stream, 1, 1) for key in order}
    cutoff = min(first.values()) * 1.15
    second = {}
    for key in reversed(order):
        count = repeats if first[key] <= cutoff or key == DEFAULT else 1
        second[key] = measure(benches[key], key, stream, 0, count)
    return {key: [first[key], second[key]] for key in keys}


def tune_output(case, output, dtype, inputs, weights, args):
    idx, m, n, k, g = (case[key] for key in ("ID", "M", "N", "K", "G"))
    a, b, sa, sb = inputs
    buf = torch.empty((g, m * n + 32), device="cuda", dtype=dtype)
    c = buf[:, 16:-16].view(g, m, n)
    ref = torch.empty_like(c, memory_format=torch.contiguous_format)
    desc = build_descriptors(a, c, sa, sb, output)
    stream = torch.cuda.current_stream().cuda_stream
    legal = candidate_list(weights, desc)
    plans, benches = {}, {}
    try:
        # The same output and prepared weights are shared across candidates.
        plan = plans[DEFAULT] = make_plan(weights, desc, DEFAULT)
        default_key = selection(plan)
        rejected = C.c_void_p()
        status = lib.ck_tile_int8_nt_tuning_plan_create_v1(
            weights, desc, g, 0xFFFFFFFF, 0, C.byref(rejected))
        assert status == -5 and not rejected.value
        for key in legal:
            if key != default_key:
                plans[key] = make_plan(weights, desc, key)
        for key, plan in plans.items():
            benches[key] = lib.cached_i8_bench_create(plan)
            assert benches[key], lib.cached_i8_bench_error().decode()
        torch.cuda.synchronize()
        keys = list(plans)
        screen = screen_samples(benches, keys, stream, args.screen_repeat, (idx + output) % 2)

        for i in range(g):
            bf = b[i].float()
            factor = sa[i] * sb[i]
            for lo in range(0, m, 128):
                hi = min(lo + 128, m)
                ref[i, lo:hi] = ((a[i, lo:hi].float() @ bf.T) * factor).to(dtype)
            del bf
        valid = []
        checks = {}
        for key in keys:
            buf.fill_(117)
            ok(lib.ck_tile_int8_nt_cached_run_v1(plans[key], stream))
            exact = torch.equal(c, ref)
            guards = bool((buf[:, :16] == 117).all() & (buf[:, -16:] == 117).all())
            checks[key] = dict(numeric_pass=exact, guards_pass=guards)
            if not guards:
                raise RuntimeError(("output guard changed", idx, output, key))
            if exact:
                valid.append(key)
        assert DEFAULT in valid, ("default correctness failure", idx, output)
        ranked = sorted(valid, key=lambda key: statistics.median(screen[key]))
        fastest_screen = statistics.median(screen[ranked[0]])
        finalists = [key for i, key in enumerate(ranked)
                     if i < args.finalists or statistics.median(screen[key]) <= fastest_screen * 1.015]
        if DEFAULT not in finalists:
            finalists.append(DEFAULT)
        formal = bench_samples(
            benches, finalists, stream, 4, args.confirm_repeat, 3, idx + output + 1)
        final_rank = sorted(finalists, key=lambda key: statistics.median(formal[key]))
        return {
            "default": list(default_key), "enumerated": len(legal),
            "executed": len(plans), "numeric_failures": len(plans) - len(valid),
            "winner": list(default_key if final_rank[0] == DEFAULT else final_rank[0]),
            "winner_ms": statistics.median(formal[final_rank[0]]),
            "default_ms": statistics.median(formal[DEFAULT]),
            "runner_up": list(default_key if final_rank[1] == DEFAULT else final_rank[1])
            if len(final_rank) > 1 else None,
            "candidates": [
                dict(
                    **legal.get(default_key if key == DEFAULT else key, {}),
                    key=list(default_key if key == DEFAULT else key),
                    default=(key == DEFAULT),
                    screen_ms=screen[key], confirm_ms=formal.get(key),
                    **checks[key],
                )
                for key in keys
            ],
        }
    finally:
        for bench in benches.values():
            if bench:
                lib.cached_i8_bench_destroy(bench)
        for plan in plans.values():
            if plan.value:
                ok(lib.ck_tile_int8_nt_cached_plan_destroy_v1(plan))


def tune_case(case, args):
    idx, m, n, k, g = (case[key] for key in ("ID", "M", "N", "K", "G"))
    assert k % 16 == 0 and n % 4 == 0
    torch.manual_seed(20261104 + idx)
    a = torch.randint(-8, 9, (g, m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 9, (g, n, k), dtype=torch.int8, device="cuda")
    sa = torch.rand(g, dtype=torch.float32, device="cuda") * 0.008 + 0.002
    sb = torch.rand((g, n), dtype=torch.float32, device="cuda") * 0.008 + 0.002
    inputs = (a, b, sa, sb)
    positions = [torch.arange(256, device="cuda") * (x.numel() - 1) // 255 for x in inputs]
    saved = [x.reshape(-1)[p].clone() for x, p in zip(inputs, positions)]
    desc = (Weight * g)(*[init(Weight, data=x.data_ptr(), n=n, k=k, row_stride=k) for x in b])
    weights = C.c_void_p()
    stream = torch.cuda.current_stream().cuda_stream
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        begin.record()
        ok(lib.ck_tile_int8_nt_cached_weights_create_v1(desc, g, stream, C.byref(weights)))
        end.record()
        end.synchronize()
        result = dict(**case, weight_prepare_ms=begin.elapsed_time(end),
                      packed_bytes=weight_info(weights).packed_bytes, outputs={})
        order = OUTPUTS if idx % 2 else reversed(OUTPUTS)
        for code, kind, dtype in order:
            result["outputs"][kind] = tune_output(case, code, dtype, inputs, weights, args)
        assert all(torch.equal(x.reshape(-1)[p], old) for x, p, old in zip(inputs, positions, saved))
        return result
    finally:
        if weights.value:
            ok(lib.ck_tile_int8_nt_cached_weights_destroy_v1(weights))


def export(path, cases):
    with path.open("w", newline="") as out:
        columns = ("ID", "phase", "M", "N", "K", "G", "output", "weight_prepare_ms", "packed_bytes",
                   "enumerated", "executed", "numeric_failures", "default", "winner", "runner_up",
                   "default_ms", "winner_ms", "winner_TOPS", "speedup", "saved_us")
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            for output, result in case["outputs"].items():
                row = {key: case[key] for key in columns if key in case}
                row.update({key: result[key] for key in columns if key in result})
                m, n, k, g = [case[key] for key in ("M", "N", "K", "G")]
                row.update(output=output, winner_TOPS=2*m*n*k*g / result["winner_ms"] / 1e9,
                           speedup=result["default_ms"] / result["winner_ms"],
                           saved_us=(result["default_ms"] - result["winner_ms"]) * 1000)
                writer.writerow(row)


def run(args, cases, manifest):
    path = args.output / "manifest.json"
    if args.resume:
        assert json.loads(path.read_text()) == manifest, "Tuning configuration changed"
    else:
        atomic(path, manifest)
    cases_path = args.output / "cases"
    cases_path.mkdir(exist_ok=args.resume)
    records = []
    for case in cases:
        result_path = cases_path / f"{case['ID']}.json"
        if args.resume and result_path.exists():
            result = json.loads(result_path.read_text())
            assert all(result[key] == val for key, val in case.items())
        else:
            atomic(args.output / "active.json", dict(case=case, completed=len(records)))
            start = time.monotonic()
            result = tune_case(case, args)
            atomic(result_path, result)
            torch.cuda.empty_cache()
            print(case["ID"], round(time.monotonic()-start, 2),
                  {out: (r["enumerated"], round(r["default_ms"]/r["winner_ms"], 4))
                   for out, r in result["outputs"].items()}, flush=True)
        records.append(result)
    assert snapshot() == manifest["build"]
    output_csv = args.output / "results.csv"
    temp_csv = output_csv.with_suffix(".tmp")
    export(temp_csv, records)
    temp_csv.replace(output_csv)
    atomic(args.output / "done.json", dict(shapes=len(records),
        executed=sum(r["executed"] for case in records for r in case["outputs"].values()),
        numeric_failures=sum(r["numeric_failures"] for case in records for r in case["outputs"].values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", type=Path, default=EXAMPLE / "benchmarks/data/targets_max_all_1191.csv")
    parser.add_argument("--ids", type=int, nargs="+", help="Limit to the specified shape IDs")
    parser.add_argument("--screen-repeat", type=int, default=3)
    parser.add_argument("--confirm-repeat", type=int, default=10)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--pci", help="Assert the selected physical PCI address")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert min(args.screen_repeat, args.confirm_repeat, args.finalists) > 0
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    assert torch.cuda.device_count() == 1, "Expose exactly one target device"
    prop = torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.startswith("gfx936")
    bus = C.create_string_buffer(64)
    lib.hipDeviceGetPCIBusId.argtypes = [C.c_void_p, C.c_int, C.c_int]
    lib.hipDeviceGetPCIBusId.restype = C.c_int
    assert lib.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
    pci = bus.value.decode()
    assert not args.pci or pci.lower() == args.pci.lower()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with args.shapes.open() as f:
        rows = [{**{k: int(r[k]) for k in ("ID", "M", "N", "K", "G")}, "phase": r["phase"]}
                for r in csv.DictReader(f)]
    ids = set(args.ids) if args.ids else {r["ID"] for r in rows}
    cases = [r for r in rows if r["ID"] in ids]
    assert len(cases) == len(ids) and cases
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(build=snapshot(), shapes_sha256=digest(args.shapes),
                    runner_sha256=digest(Path(__file__)), ids=sorted(ids), pci=pci,
                    device=str(prop), screen_repeat=args.screen_repeat,
                    confirm_repeat=args.confirm_repeat, finalists=args.finalists)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, cases, manifest)


if __name__ == "__main__":
    main()
