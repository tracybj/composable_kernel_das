"""Trace actual CPU selector-test inputs into the future heldout exclusions.

Only an isolated shadow header gains runtime logging. Its constexpr behavior
and selector statements remain unchanged. No measured/build source is edited.
"""
from pathlib import Path
import datetime,hashlib,json,os,resource,subprocess
ROOT=Path(__file__).resolve().parent
ISO=ROOT/'isolated_resource_build'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-4:])
    prep=json.loads((ISO/'preparation.json').read_text());src=Path(prep['source'])
    folder=ISO/'cpu_geometry_trace';folder.mkdir(exist_ok=False)
    relative='ck_tile/ops/gemm/grouped_gemm_selector.hpp';original=src/'include'/relative
    text=original.read_text();anchor='''constexpr GroupedGemmSelection select_grouped_gemm_candidate(
    const GroupedGemmProblem& problem,
    const GroupedGemmSelectorPolicy& policy = {})
{'''
    assert text.count(anchor)==1
    trace='''
    if(!__builtin_is_constant_evaluated())
        std::cout << "CPU_USED_GEOMETRY " << problem.m.extent << ' '
                  << problem.n.extent << ' ' << problem.k.extent << ' '
                  << problem.group_count << '\\n';'''
    instrumented='#include <iostream>\n'+text.replace(anchor,anchor+trace,1)
    header=folder/'include'/relative;header.parent.mkdir(parents=True);header.write_text(instrumented)
    commands=[];used=set();counts={}
    for name in ['grouped_gemm_decode_mainline_selector_test','grouped_gemm_decode_depth8_selector_test','grouped_gemm_selector_test']:
        source=src/'example_hcu/ck_tile/19_grouped_gemm'/(name+'.cpp');exe=folder/name
        command=['g++','-O2','-std=c++17','-I'+str(folder/'include'),'-I'+str(src/'include'),str(source),'-o',str(exe)]
        result=subprocess.run(command,capture_output=True,text=True)
        (folder/(name+'_compile.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr[-5000:]
        run=subprocess.run([str(exe)],capture_output=True,text=True)
        (folder/(name+'_run.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0 and 'PASS' in run.stdout
        geometries=[]
        for line in run.stdout.splitlines():
            if line.startswith('CPU_USED_GEOMETRY '):
                shape=tuple(map(int,line.split()[1:]));assert len(shape)==4
                geometries.append(shape)
                if all(x>0 for x in shape):used.add(shape)
        assert geometries,'No actual selector invocations were traced'
        counts[name]=len(geometries);commands.append(dict(command=command,source_sha256=sha(source),compile_rc=result.returncode,run_rc=run.returncode))
    assert original.read_text()==text
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_only=True,GPU=False,
        purpose='Exclude every positive geometry actually used in these CPU selector-design regressions from future heldout draws',
        selector_original_sha256=sha(original),shadow_header_sha256=sha(header),
        measured_and_isolated_build_sources_unchanged=True,tests_passed=True,counts=counts,
        unique_geometries=len(used),cases=[dict(zip(['M','N','K','G'],x)) for x in sorted(used)],
        commands=commands,not_performance_evidence=True,
        required_followup='After current pinned controllers stop, add this exact file to final_confirmation.previously_used; do not generate a new seed early')
    path=ISO/'cpu_geometry_inventory.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print('CPU_RESOURCE_GEOMETRY_INVENTORY_PASS',len(used),counts,'NO_GPU',flush=True)

if __name__=='__main__':main()
