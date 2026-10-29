"""CPU-only checks of frozen contract, missing CSV rows and resume validation."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile

path = Path(__file__).with_name('validate_all_1178.py')
spec = importlib.util.spec_from_file_location('cached_acceptance', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cases = module.contract()
frozen = module.snapshot()

# Run the public plan-only command, proving it needs neither Torch nor HIP.
sys.argv = [str(path), '--plan-only']
with contextlib.redirect_stdout(io.StringIO()) as captured:
    runpy.run_path(str(path), run_name='__main__')
report = json.loads(captured.getvalue())
assert report['status'] == 'CPU_CONTRACT_PASS_GPU_NOT_RUN'
assert report['shapes'] == 1178 and report['snapshot'] == frozen
assert 'torch' not in sys.modules and 'bindings' not in sys.modules

t = cases[0]
rec = {k: int(t[k]) for k in ['ID', 'M', 'N', 'K', 'G']}
rec.update(phase=t['phase'], library_sha256=frozen['api_library'],
           dtype='INT8xINT8->INT32_no_scale', layout='NT',
           BLAS_target_ms=float(t['target_ms']), BLAS_target_TOPS=float(t['target_TOPS']),
           numeric_status='PASS', bad_elements=0, guard=True,
           input_samples_unchanged=True, warmup=10, repeat=50, samples=3,
           samples_ms=[1.1, 1.2, 1.3], CK_ms=1.2)
rec['CK_TOPS'] = 2*rec['G']*rec['M']*rec['N']*rec['K'] / rec['CK_ms'] / 1e9
rec['R'] = rec['CK_TOPS'] / float(t['target_TOPS'])
module.verify_prefix(cases, [rec], frozen['api_library'])
for changes in [dict(ID=rec['ID']+1), dict(M=rec['M']+1),
                dict(numeric_status='PENDING'), dict(library_sha256='wrong'),
                dict(bad_elements=1), dict(guard=False),
                dict(input_samples_unchanged=False), dict(repeat=5),
                dict(samples_ms=[1.1]), dict(CK_TOPS=rec['CK_TOPS']*2),
                dict(R=rec['R']*2), dict(layout='NN'), dict(dtype='BF16'),
                dict(BLAS_target_TOPS=float(t['target_TOPS'])*2),
                dict(BLAS_target_ms=float(t['target_ms'])*2)]:
    bad = copy.deepcopy(rec)
    bad.update(changes)
    try:
        module.verify_prefix(cases, [bad], frozen['api_library'])
    except AssertionError:
        pass
    else:
        raise AssertionError(('Corrupt resume was accepted', changes))

with tempfile.TemporaryDirectory() as temporary:
    out = Path(temporary)
    module.export(out, cases, [], frozen['api_library'])
    rows = list(csv.DictReader((out / 'results.csv').open()))
    assert len(rows) == 1178
    assert all(x['numeric_status'] == 'PENDING' and x['CK_ms'] == '' for x in rows)
    module.export(out, cases, [rec], frozen['api_library'])
    rows = list(csv.DictReader((out / 'results.csv').open()))
    assert rows[0]['numeric_status'] == 'PASS' and rows[0]['CK_ms'] == '1.2'
    assert sum(x['numeric_status'] == 'PENDING' for x in rows) == 1177
    assert all(all(x[k] == t[k] for k in ['ID', 'phase', 'M', 'N', 'K', 'G'])
               for x, t in zip(rows, cases))

print(json.dumps(dict(status='PASS', scope='CPU harness checks,not GPU results',
                      original_shapes=1178, corrupt_resume_rejections=15,
                      pending_rows_not_marked_pass=True, torch_not_imported=True,
                      snapshot=frozen), indent=2))
