#!/usr/bin/env python3
"""Replay the frozen BLAS algorithm; diagnostic only, no retuning or rebuild."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

TUNING = Path('/workspace/Grouped_Geem/blaslt_tuning_tools')
BASELINE = TUNING / 'results/bf16_nn_compare_20261027_191315/groupgemm'
sys.path.insert(0, str(TUNING))
from tuning_common import Desc, Native, restore_inputs, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, required=True, choices=[76, 681, 633, 358])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--validate', type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    def phase(name):
        print(f'SETUP phase={name} seconds={time.monotonic()-started:.3f}', flush=True)
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4'
    phase('before_torch')
    import torch
    phase('after_torch')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    cfg = json.loads((BASELINE / 'run_config.json').read_text())
    case = next(c for c in cfg['cases'] if c['id'] == args.id)
    with (BASELINE / 'results.csv').open(encoding='utf-8-sig') as f:
        frozen = next(c for c in csv.DictReader(f) if int(c['ID']) == args.id)
    cfg.update(gpu=4, binary=str(TUNING / 'build/libgroupgemm_tune.so'))
    assert sha(cfg['blas_library']) == cfg['blas_sha256']
    g, m, n, k = (case[x] for x in ('G', 'M', 'N', 'K'))
    a = torch.empty((g, m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.empty((g, k, n), device='cuda', dtype=torch.bfloat16)
    seeds = tuple((cfg['seed'] + args.id + i) % (1 << 64) for i in range(2))
    restore_inputs((a, b), seeds)
    phase('inputs_ready')
    raw = torch.full((g, m*n+32), 117, device='cuda', dtype=torch.bfloat16)
    c = raw[:, 16:-16].view(g, m, n)
    c.fill_(float('nan'))
    descs = (Desc * g)(*[Desc(a[i].data_ptr(), b[i].data_ptr(), c[i].data_ptr()) for i in range(g)])
    native = Native(cfg, case, descs)
    phase('native_ready')
    try:
        candidate = next(x for x in native.candidates if x['algorithm_id'] == int(frozen['algorithm_id']))
        assert candidate['kernel_name'] == frozen['kernel_name']
        assert candidate['workspace_bytes'] == int(frozen['workspace_bytes'])
        print('REPLAY ' + json.dumps(dict(id=args.id, gpu=4, case=case, candidate=candidate,
              bridge_sha256=sha(cfg['binary']), blas_sha256=cfg['blas_sha256'],
              device=str(torch.cuda.get_device_properties(0)))), flush=True)
        torch.cuda.synchronize()
        index = candidate['candidate_index']
        times = [native.time(index, descs, args.warmup if i == 0 else 0, args.repeat)
                 for i in range(args.samples)]
        result = dict(id=args.id, gpu=4, backend='blas', repeat=args.repeat,
                      warmup=args.warmup, samples=args.samples, samples_ms=times,
                      median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times),
                      cv=statistics.pstdev(times)/statistics.mean(times),
                      algorithm_id=candidate['algorithm_id'], kernel_name=candidate['kernel_name'])
        if args.validate:
            phase('before_reference')
            # Inputs are multiples of 1/8 and K<=2048, so these FP32 dot
            # products are exact. CPU reference avoids loading a second GPU
            # BLAS backend solely for validation of this diagnostic harness.
            reference = torch.bmm(a.cpu().float(), b.cpu().float()).to(torch.bfloat16)
            actual = c.cpu()
            guards = raw.cpu()
            result['reference'] = 'cpu_fp32_bmm'
            result['checked'] = c.numel()
            result['failed'] = int((actual != reference).sum().item())
            result['max_abs'] = float((actual.float() - reference.float()).abs().max().item())
            result['guards_ok'] = bool(((guards[:, :16] == 117).all() & (guards[:, -16:] == 117).all()).item())
            assert result['failed'] == 0 and result['max_abs'] == 0 and result['guards_ok'], result
        print('RESULT ' + json.dumps(result), flush=True)
    finally:
        native.close()


if __name__ == '__main__':
    main()
