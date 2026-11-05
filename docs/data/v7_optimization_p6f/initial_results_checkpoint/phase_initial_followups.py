"""Complete only the frozen finite boundary, causal and profile followups."""
import subprocess,sys,time
import runtime as r
for suite,plan in [('initial_extra_accuracy','extra_accuracy_frozen.json'),
                   ('initial_isolation_accuracy','isolation_accuracy_frozen.json'),
                   ('initial_isolation_perf','isolation_perf_frozen.json'),
                   ('initial_decode_profiles_filterfix','decode_profiles_filterfix_frozen.json'),
                   ('initial_isolation_profiles_filterfix','isolation_profiles_filterfix_frozen.json')]:
    path=r.ROOT/(suite+'_controller.log');assert not path.exists()
    cmd=[sys.executable,'-u',str(r.ROOT/'campaign.py'),suite,'--plan',str(r.ROOT/plan)]
    with path.open('w') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    r.append(r.ROOT/'initial_followup_steps.jsonl',dict(at=r.stamp(),suite=suite,rc=result.returncode,log=str(path)))
    print('INITIAL_FOLLOWUP',suite,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
