"""Integrate the one preregistered phase-order choice and build, CPU only."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,shutil,subprocess,time
import plan,runtime as r,shader_audit
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
HERE=ROOT/'drafts/resource_interleave'

def main():
    r.no_core();session=r.read(ROOT/'session.json')
    assert time.time()+3600<session['gpu_deadline_epoch']
    assert r.read(ROOT/'resource_controller_status.json')['state']=='complete'
    assert r.read(ROOT/'resource_source_commit.json')['commit']=='b78c07bf2dda251121987d7c74b1b1efbda338de'
    assert r.read(ROOT/'profile_delivery_batch8.json')['captures']==10
    choice=r.read(HERE/'probe_resources.json');assert choice['ok'] and choice['choice']=='all_compute_first'
    assert r.read(HERE/'probe_schedule_audit.json')['ok']
    overlay=r.read(HERE/'selected_overlay.json')
    assert all(plan.sha(REPO/n)==sha for n,sha in overlay['before_sha256'].items())
    patch=HERE/'selected_integration.patch';assert plan.sha(patch)==overlay['patch_sha256']
    subprocess.run(['git','apply','--check','--whitespace=error-all',str(patch)],cwd=REPO,check=True)
    folder=ROOT/'compile/build9';assert not folder.exists()
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    folder.mkdir(parents=True);backup=folder/'previous';backup.mkdir()
    old=r.read(ROOT/'frozen_plan.json');assert old['phase']=='h3_resource_discovery'
    assert plan.sha(plan.BENCH)==old['benchmark_sha256']=='eb02cd5d383669c7f12916f2dc11910c27f827b8837e59bfd8974426e4761fd6'
    for name,sha in old['worktree_source_sha256'].items():
        assert plan.sha(REPO/name)==sha
        target=backup/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
    for name in old['runner_sha256']:
        source=ROOT/name;target=backup/'runners'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(ROOT/'frozen_plan.json',backup/'frozen_plan.json')
    names=['tile_example_grouped_gemm_device_args_benchmark','tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke','tile_example_grouped_gemm_selector_test','tile_example_grouped_gemm_decode_depth8_selector_test','tile_example_grouped_gemm_decode_mainline_selector_test']
    for name in names:shutil.copy2(REPO/'build/bin'/name,backup/name)
    kernels=shader_audit.inventory(plan.BENCH);assert len(kernels)==1710
    r.write(folder/'reference_shaders.json',dict(binary_sha256=plan.sha(plan.BENCH),kernels=kernels))
    subprocess.run(['git','apply','--whitespace=error-all',str(patch)],cwd=REPO,check=True)
    assert all(plan.sha(REPO/n)==sha for n,sha in overlay['after_sha256'].items())
    sources=dict(old['worktree_source_sha256']);sources.update(overlay['after_sha256'])
    for name in sources:sources[name]=plan.sha(REPO/name)
    r.write(folder/'source_sha256.json',sources)
    r.write(ROOT/'active_build.json',dict(folder='compile/build9',at=r.stamp(),script='build_phase_factor.py',phase='compile_selected_compute_first',GPU=False))
    r.write(ROOT/'phase_hypothesis.json',dict(at=r.stamp(),registered_CPU_hypothesis_sha256=plan.sha(HERE/'schedule_hypothesis.json'),
        probe_resources_sha256=plan.sha(HERE/'probe_resources.json'),actual_CFG_audit_sha256=plan.sha(HERE/'probe_schedule_audit.json'),
        selected_schedule='All waves compute current MMAC before unchanged ready/read/prefetch',
        reason='Fixed CPU rule chose lower summed VGPR count:493->467. WideM32 83->73,WideM64 128->112; compact63/86/133 unchanged. Resources/VM/LG/producer-read-barrier order and K accumulation unchanged. GPU benefit remains untested.',
        previous_discovery_sha256=plan.sha(ROOT/'resource_discovery_results.json'),previous_profiles_sha256=plan.sha(ROOT/'resource_profile_analysis/profile_analysis.json'),
        benchmark_before_sha256=old['benchmark_sha256'],new_instances=[6026,6027],variants=overlay['variants'],
        main_ownership_architectures=3,short_ownership_architectures=2,not_fourth_architecture=True,
        original_deadlines_unchanged=True,original50_stable_and30_attributable_targets_unchanged=True,
        new_GPU_plan_required_before_any_GPU=True,adopted=False))
    os.sched_setaffinity(0,{124,125,126,127});os.nice(10)
    commands=[['cmake','-S',str(REPO),'-B',str(REPO/'build'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],
        ['cmake','--build',str(REPO/'build'),'--target',*names,'-j4']]
    results=[]
    for i,command in enumerate(commands):
        assert time.time()+60<session['gpu_deadline_epoch']
        assert all(plan.sha(REPO/n)==sha for n,sha in sources.items())
        memory=int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1]);assert memory>=96*1024*1024
        record=dict(at=r.stamp(),command=command,GPU=False,stage=i)
        r.write(folder/f'{i}_command.json',record);start=time.monotonic()
        env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4')
        with (folder/f'{i}_build.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            record['pid']=child.pid;r.write(folder/'live.json',record);rc=child.wait()
        record.update(rc=rc,seconds=time.monotonic()-start,finished_at=r.stamp());results.append(record);r.write(folder/'results.json',results)
        print('PHASE_BUILD_STAGE',i,'rc',rc,'seconds',record['seconds'],flush=True)
        if rc:raise SystemExit(rc)
    assert all(plan.sha(REPO/n)==sha for n,sha in sources.items())
    r.write(folder/'completed.json',dict(at=r.stamp(),CPU_only=True,GPU=False,benchmark_sha256=plan.sha(plan.BENCH),source_sha256=sources))
    print('PHASE_FACTOR_CPU_BUILD_PASS',plan.sha(plan.BENCH),flush=True)

if __name__=='__main__':main()
