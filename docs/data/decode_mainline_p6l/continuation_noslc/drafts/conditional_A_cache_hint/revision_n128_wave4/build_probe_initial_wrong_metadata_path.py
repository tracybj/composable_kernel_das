"""CPU-only dual-architecture feasibility probe, never executes a kernel."""
from pathlib import Path
import datetime,hashlib,json,os,re,resource,subprocess,sys,time
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    assert time.time()+180<read(ROOT/'session.json')['new_features_deadline_epoch']
    assert read(ROOT/'wave4_discovery_analysis.json')['complete_phase']
    assert read(ROOT/'wave4_profile_repair_status.json')['state']=='stopped'
    assert not (ROOT/'environment_recovery_required.json').exists()
    assert int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])>=96*1024*1024
    previous=ROOT/'drafts/conditional_n128_wave4'
    cmd=read(previous/'probe_command.json')['command']
    cmd=[x.replace(str(previous/'probe.cpp'),str(D/'probe.cpp')).replace(str(previous/'probe.o'),str(D/'probe.o')) for x in cmd]
    cmd=[('-I'+str(D/'provider_overlay/include')) if x=='-I'+str(previous/'provider_overlay/include') else x for x in cmd]
    assert cmd[:6]==['taskset','-c','120-123','nice','-n','10']
    assert cmd[7]=='-I'+str(D/'provider_overlay/include')
    header=D/'provider_overlay/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_n256.hpp'
    record=dict(at=stamp(),command=cmd,GPU=False,CPU_only=True,header_sha256=sha(header),probe_sha256=sha(D/'probe.cpp'))
    (D/'probe_command.json').write_text(json.dumps(record,indent=2)+'\n');start=time.monotonic()
    with (D/'probe_build.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        (D/'probe_live.json').write_text(json.dumps(dict(**record,pid=child.pid),indent=2)+'\n');rc=child.wait()
    record.update(rc=rc,seconds=time.monotonic()-start,finished_at=stamp());(D/'probe_result.json').write_text(json.dumps(record,indent=2)+'\n')
    assert rc==0,'CPU probe failure retained; no automatic retry'
    sys.path.insert(0,str(ROOT));import shader_audit
    kernels=shader_audit.inventory(D/'probe.o');assert len(kernels)==6
    assert all(x['metadata'].get('.private_segment_fixed_size',0)==0 and x['metadata'].get('.vgpr_spill_count',0)==0 and x['metadata'].get('.sgpr_spill_count',0)==0 for x in kernels.values())
    result=dict(at=stamp(),ok=True,CPU_only=True,GPU=False,binary_sha256=sha(D/'probe.o'),source_sha256=sha(header),kernels=kernels,not_GPU_precision_or_timing_evidence=True)
    (D/'probe_resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print('A_NOSLC_CPU_PROBE_PASS',json.dumps({k:v['metadata'] for k,v in kernels.items()}),flush=True)
if __name__=='__main__':main()
