"""CPU-only subprocess fixture. Does not load HIP or issue GPU commands."""
import argparse
import json
import os

ap=argparse.ArgumentParser();ap.add_argument('--case-json',required=True);ap.add_argument('--kind',required=True)
args=ap.parse_args();case=json.loads(args.case_json)
env={k:v for k,v in os.environ.items() if k.startswith('CK_TILE_GROUPED_GEMM_')}
assert env==case['env'],(env,case['env'])
assert os.environ['HIP_VISIBLE_DEVICES']==os.environ['CUDA_VISIBLE_DEVICES']=='4'
print('FAKE_ENV '+json.dumps(env))
mode='off' if args.kind=='accuracy' else 'on';guard=16 if args.kind=='accuracy' else 0
print(f"CONFIG arch=gfx936 dtype=bf16 layout=NN groups={case['G']} m={case['M']} n={case['N']} k={case['K']} variable_axis=none timing={mode} warmup=10 samples=30 launches_per_sample=1 include_args_build=0 num_cu=0 blas_streams=1 output_redzone_elements={guard}")
print('SELECT instance_id='+str(case['expected']))
print('ACCURACY backend=ck repeat=0 checked=257 failed=0 max_abs=0 max_rel=0')
if args.kind=='accuracy':
    for backend,count in [('ck_guard',0),('blas_guard',0),('ck_redzone',32),('blas_redzone',32),('ck_vs_blas',case['M']*case['N']*case['G'])]:
        print(f'ACCURACY backend={backend} repeat=0 checked={count} failed=0 max_abs=0')
else:
    print('PERF backend=ck_device_args median_ms=0.1 min_ms=0.09 max_ms=0.11 cv=0.01')
print('SUMMARY result=PASS')
