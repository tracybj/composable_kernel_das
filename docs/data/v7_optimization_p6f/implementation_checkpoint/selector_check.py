from pathlib import Path
import hashlib,json,subprocess
import runtime as r
base=r.ROOT/'selector_baseline.hpp'
base.write_bytes(subprocess.check_output(['git','show',r.session()['head']+':include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'],cwd=r.REPO))
source=Path('/workspace/v7_coverage/p6e/selector_equivalence.cpp')
results=[]
for name,header in [('baseline',str(base)),('working','ck_tile/ops/gemm/grouped_gemm_selector.hpp')]:
    binary=r.ROOT/('selector_equivalence_'+name)
    cmd=['g++','-std=c++17','-O2','-I'+str(r.REPO/'include'),'-DSELECTOR_HEADER="'+header+'"',str(source),'-o',str(binary)]
    subprocess.run(cmd,check=True)
    p=subprocess.run([str(binary)],stdout=subprocess.PIPE,check=True)
    (r.ROOT/('selector_equivalence_'+name+'.csv')).write_bytes(p.stdout);results.append(p.stdout)
assert results[0]==results[1]
cmd=['g++','-std=c++17','-O2','-I'+str(r.REPO/'include'),str(r.ROOT/'selector_cache_cpu.cpp'),'-o',str(r.ROOT/'selector_cache_cpu')]
subprocess.run(cmd,check=True);p=subprocess.run([str(r.ROOT/'selector_cache_cpu')],stdout=subprocess.PIPE,check=True,text=True)
print(p.stdout,end='')
record=dict(existing_selector_cases=len(results[0].splitlines()),byte_identical=True,baseline_head=r.session()['head'],cache_checks_stdout=p.stdout,working_header_sha256=r.sha(r.REPO/'include/ck_tile/ops/gemm/grouped_gemm_selector.hpp'))
(r.ROOT/'selector_cache_audit.json').write_text(json.dumps(record,indent=2));print('EXISTING_SELECTOR_EQUIVALENCE_PASS',record['existing_selector_cases'])
