"""Strict, GPU-free parsing and selector checks for recorded BF16 NN recipes."""
import csv
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / 'build/bin/tile_example_grouped_gemm_device_args_benchmark'
BLAS = Path('/workspace/Grouped_Geem/hipblaslt-install/lib/libhipblaslt.so')
SHAPES = [Path('/workspace/shapes_1178.csv'), Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv')]
SHAPE_SHA = '5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863'
PREFIX = 'CK_TILE_GROUPED_GEMM_'
V7 = PREFIX + 'ENABLE_BF16_V7_NN'
DEPTH8 = PREFIX + 'ENABLE_BF16_NN_DECODE_DEPTH8'
SELECTOR = REPO / 'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'
FLAGS = {V7: {'0', '1'}, PREFIX+'DISABLE_BF16_V7_NN': {'0', '1'}}
FLAGS[DEPTH8] = {'0', '1'}
FLAGS[PREFIX+'DISABLE_BF16_NN_DECODE_DEPTH8'] = {'0', '1'}
for suffix, values in [('M01_8', {'0', '1'}), ('SMALL_M', {'0', '64', '128'}),
                       ('A_NOSLC', {'0', '1'}), ('CACHE_WALK', {'0', '1', '5', '8'}),
                       ('N_TAIL', {'0', '1', '2'}), ('CACHE_GRID', {'0', '1'}),
                       ('SMALL_M_EXPERIMENT', {str(i) for i in range(8)})]:
    FLAGS[V7+'_'+suffix] = values
    FLAGS[PREFIX+'DISABLE_BF16_V7_NN_'+suffix] = {'0', '1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def configure_paths(blas_lib=None, shape_files=None):
    """Configure caller-owned inputs without copying or changing their contents."""
    global BLAS, SHAPES
    if blas_lib is not None:
        BLAS = Path(blas_lib).expanduser().absolute()
    if shape_files is not None:
        if not shape_files:
            raise ValueError('At least one shape file is required')
        SHAPES = [Path(p).expanduser().absolute() for p in shape_files]


def runtime_paths():
    return dict(blas_lib=str(BLAS), shape_files=[str(p) for p in SHAPES])


def dependencies():
    """Fingerprint linked DSOs and the explicitly loaded BLAS dependencies."""
    paths={str(BLAS.resolve())}
    logs={}
    for binary in [BENCH,BLAS]:
        p=subprocess.run(['ldd',str(binary)],env=environment({}),capture_output=True,text=True,check=True,timeout=30)
        logs[str(binary)]=p.stdout
        for line in p.stdout.splitlines():
            for token in line.split():
                if token.startswith('/') and Path(token).is_file():paths.add(str(Path(token).resolve()))
    for name in ['liborigami.so.1','liborigami.so','hipblaslt/library_gfx936/TensileLibrary_gfx936.co']:
        p=BLAS.parent/name
        if p.is_file():paths.add(str(p.resolve()))
    return dict(sha256={name:sha(name) for name in sorted(paths)},ldd=logs)


def validate_env(env):
    if not isinstance(env, dict):
        raise ValueError('enable_env must be an object')
    for key, value in env.items():
        if key not in FLAGS or not isinstance(value, str) or value not in FLAGS[key]:
            raise ValueError(f'Unknown switch/value: {key}={value!r}')
    return env


def environment(overrides, inherited=None):
    validate_env(overrides)
    src = os.environ if inherited is None else inherited
    env = {k: v for k, v in src.items() if not k.startswith(PREFIX)}
    for key in ['LD_PRELOAD','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:
        env.pop(key,None)
    env.update(overrides)
    env.update(HIP_VISIBLE_DEVICES='4', CUDA_VISIBLE_DEVICES='4', OMP_NUM_THREADS='4')
    env['LD_LIBRARY_PATH'] = str(BLAS.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    return env


def load_table(path, shape_files=None):
    shape_files = SHAPES if shape_files is None else shape_files
    path = Path(path).resolve()
    for source in shape_files:
        if sha(source) != SHAPE_SHA:
            raise ValueError(f'Shape hash changed: {source}')
    with shape_files[0].open(encoding='utf-8-sig', newline='') as f:
        shapes = {int(x['ID']): x for x in csv.DictReader(f)}
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    cases = []
    for row in rows:
        i = int(row['ID'])
        dims = {k: int(row[k]) for k in ['M', 'N', 'K', 'G']}
        if i not in shapes or any(dims[k] <= 0 or dims[k] != int(shapes[i][k]) for k in dims):
            raise ValueError(f'Shape mismatch ID{i}')
        phase = 'decode' if dims['M'] <= 256 else 'prefill'
        if row['phase'] != phase:
            raise ValueError(f'Phase mismatch ID{i}')
        prior = float(row['CK_ms']); blas = float(row['frozen_BLAS_ms'])
        if not all(math.isfinite(x) and x > 0 for x in [prior, blas]):
            raise ValueError(f'Nonpositive/nonfinite timing ID{i}')
        if not math.isclose(float(row['R_frozen']), blas / prior, rel_tol=1e-10):
            raise ValueError(f'Historical R mismatch ID{i}')
        env = validate_env(json.loads(row['enable_env']))
        if env.get(V7) != '1':
            raise ValueError(f'Missing explicit V7 enable ID{i}')
        if int(row['BLAS_gpu']) != (0 if i < 1179 else 4) or '20' not in row['BLAS_protocol']:
            raise ValueError(f'Frozen Native protocol mismatch ID{i}')
        cases.append(dict(ID=i, **dims, phase=phase, expected=int(row['instance_id']), env=env,
            historical_expected=int(row['instance_id']),historical_env=copy.deepcopy(env),
            historical_scheme=row['scheme'],configured_scheme=row['scheme'],candidate_overlay_applied=False,
            accuracy_repeats=1,
            historical_CK_ms=prior, historical_R=blas/prior, frozen_BLAS_ms=blas,
            BLAS_gpu=int(row['BLAS_gpu']), BLAS_protocol=row['BLAS_protocol'],
            BLAS_source=row['BLAS_source'], BLAS_high_span_warning=row['BLAS_high_span_warning'],
            historical_source_type=row['source_type'], historical_source=row['source'],
            historical_binary_sha256=row['binary_sha256'], historical_quality=row['quality']))
    cases.sort(key=lambda x: x['ID'])
    if [x['ID'] for x in cases] != list(range(1, 1194)):
        raise ValueError('Expected unique contiguous IDs1..1193')
    if sum(x['phase'] == 'prefill' for x in cases) != 312:
        raise ValueError('Expected312prefill/881decode')
    return dict(csv_path=str(path), csv_sha256=sha(path), shape_sha256=SHAPE_SHA, cases=cases)


def depth8_scope(case):
    return 1 <= case['M'] <= 32 and (case['N'], case['K'], case['G']) == (256, 512, 64)


def with_depth8_overlay(table):
    table = copy.deepcopy(table)
    hits = []
    for case in table['cases']:
        if depth8_scope(case):
            case['env'][DEPTH8] = '1'
            case['expected'] = 6003
            case['configured_scheme'] = 'decode_a_reg_b_private_lds_depth8'
            case['candidate_overlay_applied'] = True
            case['accuracy_repeats'] = 3
            hits.append(case['ID'])
    if len(hits) != 6:
        raise ValueError('Depth8 predicate must cover exactly six original shapes')
    table['execution_overlay'] = dict(schema_version=1,instance_id=6003,enable_flag=DEPTH8,
        predicate='gfx936 BF16 NN packed common nonempty nonragged; 1<=M<=32,N256,K512,G64',
        matched_ids=hits,source_timings_retained_as_historical=True)
    return table


def policy_fields(case):
    e = case['env']
    enabled = e.get(V7) == '1' and e.get(PREFIX+'DISABLE_BF16_V7_NN') != '1'
    def on(s):
        return e.get(V7+'_'+s) == '1' and e.get(PREFIX+'DISABLE_BF16_V7_NN_'+s) != '1'
    sm = int(e.get(V7+'_SMALL_M', '0'))
    if e.get(PREFIX+'DISABLE_BF16_V7_NN_SMALL_M') == '1': sm = 0
    walk = int(e.get(V7+'_CACHE_WALK', '0'))
    if e.get(PREFIX+'DISABLE_BF16_V7_NN_CACHE_WALK') == '1': walk = 0
    tail = int(e.get(V7+'_N_TAIL', '0'))
    if e.get(PREFIX+'DISABLE_BF16_V7_NN_N_TAIL') == '1': tail = 0
    experiment = int(e.get(V7+'_SMALL_M_EXPERIMENT', '0'))
    if any(e.get(PREFIX+'DISABLE_BF16_V7_NN_'+s) == '1' for s in ['SMALL_M', 'SMALL_M_EXPERIMENT']): experiment = 0
    return [int(enabled), int(e.get(PREFIX+'DISABLE_BF16_V7_NN_M01_8') != '1'), sm,
            int(on('A_NOSLC')), walk, tail, experiment, int(on('CACHE_GRID')),
            int(e.get(DEPTH8)=='1' and e.get(PREFIX+'DISABLE_BF16_NN_DECODE_DEPTH8')!='1')]


def selector_check(table, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    executable = output / 'selector_probe'
    cmd = ['g++', '-O2', '-std=c++17', '-I'+str(REPO/'include'),
           str(Path(__file__).with_name('selector_probe.cpp')), '-o', str(executable)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    lines = [' '.join(map(str, [x['ID'], x['M'], x['N'], x['K'], x['G'], x['expected']] + policy_fields(x))) for x in table['cases']]
    result = subprocess.run([str(executable)], input='\n'.join(lines)+'\n', capture_output=True, text=True, check=True, timeout=30)
    actual = result.stdout.splitlines()
    if len(actual) != len(lines): raise ValueError('Selector probe output count')
    for x, line in zip(table['cases'], actual):
        i, expected, selected = map(int, line.split())
        if (i, expected, selected) != (x['ID'], x['expected'], x['expected']):
            raise ValueError(f'Selector mismatch {i}: expected{expected}, selected{selected}')
    return dict(passed=True, cpu_only=True, cases=len(lines), command=cmd,
                selector_sha256=sha(SELECTOR), output=result.stdout)


def command(case, kind):
    args = [str(BENCH), '--dtype=bf16', '--layout=NN',
            *[f'--{arg}={case[key]}' for arg, key in [('m','M'),('n','N'),('k','K'),('groups','G')]],
            '--blas-lib='+str(BLAS), '--include-args-build=0', '--validate=1',
            '--warmup=10', '--samples=30', '--launches-per-sample=1',
            '--correctness-repeats='+str(case.get('accuracy_repeats',1) if kind=='accuracy' else 1)]
    return args + (['--backend=both', '--time=0', '--output-redzone=1'] if kind == 'accuracy' else ['--backend=ck', '--time=1', '--output-redzone=0'])
