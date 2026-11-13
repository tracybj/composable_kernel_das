"""Incremental official build of a cache-policy revision of experimental6035."""
from pathlib import Path
import difflib,hashlib,json,os,re,shutil,subprocess,sys,time
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path.insert(0,str(ROOT));import plan,runtime as r,shader_audit
REPO=plan.REPO
def main():
 r.no_core();session=r.read(ROOT/'session.json');assert time.time()+1200<session['gpu_deadline_epoch']
 assert r.read(ROOT/'guard_controller_status.json')['state']=='complete'
 assert r.read(ROOT/'guard_source_commit.json')['user_untracked10_SHA_verified']
 assert r.read(ROOT/'guard_archive_commit.json')['user_worktree_preserved']
 assert r.read(D/'probe_resources.json')['ok']
 assert not any((ROOT/n).exists() for n in ['gpu_stop.json','environment_recovery_required.json','user_pause.json'])
 old=r.read(ROOT/'frozen_plan.json');assert old['benchmark_sha256']==plan.sha(plan.BENCH)
 assert all(plan.sha(REPO/n)==h for n,h in old['worktree_source_sha256'].items())
 folder=ROOT/'compile/build17';assert not folder.exists();assert shutil.disk_usage(ROOT).free>=20*1024**3
 folder.mkdir(parents=True);backup=folder/'previous';backup.mkdir()
 names=['tile_example_grouped_gemm_device_args_benchmark','tile_example_grouped_gemm_device_args_query_test','tile_example_grouped_gemm_device_args_smoke','tile_example_grouped_gemm_selector_test','tile_example_grouped_gemm_decode_depth8_selector_test','tile_example_grouped_gemm_decode_mainline_selector_test']
 for n in names:shutil.copy2(REPO/'build/bin'/n,backup/n)
 for n,h in old['worktree_source_sha256'].items():
  p=backup/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
 shutil.copy2(ROOT/'frozen_plan.json',backup/'frozen_plan.json')
 reference=shader_audit.inventory(plan.BENCH);assert len(reference)==1766
 r.write(folder/'reference_shaders.json',dict(binary_sha256=plan.sha(plan.BENCH),kernels=reference))
 header='include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_cooperative_n128_noslc.hpp'
 note=r.read(D/'mechanism_decision.json')
 assert plan.sha(REPO/header)==note['previous_source_sha256']
 assert plan.sha(D/'provider_overlay'/header)==note['candidate_source_sha256']
 shutil.copy2(D/'provider_overlay'/header,REPO/header)
 sources=dict(old['worktree_source_sha256']);sources[header]=plan.sha(REPO/header)
 assert len(sources)==26
 r.write(folder/'source_sha256.json',sources)
 for n,h in sources.items():
  p=folder/'compiled_sources'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p);assert plan.sha(p)==h
 r.write(ROOT/'active_build.json',dict(folder='compile/build17',at=r.stamp(),phase='accumulator_revision_experimental6035',GPU=False))
 os.sched_setaffinity(0,{124,125,126,127});os.nice(10)
 cmd=['cmake','--build',str(REPO/'build'),'--target',*names,'-j4']
 rec=dict(at=r.stamp(),command=cmd,GPU=False,stage=0)
 r.write(folder/'controller.json',dict(state='running',pid=os.getpid(),at=r.stamp(),GPU=False,script_sha256=plan.sha(__file__)))
 try:
  assert int(re.search(r'MemAvailable:\s+(\d+)',Path('/proc/meminfo').read_text())[1])>=96*1024*1024
  assert time.time()+60<session['gpu_deadline_epoch']
  env=os.environ.copy();env.update(HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4')
  start=time.monotonic()
  with (folder/'0_build.log').open('x') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   rec['pid']=child.pid;r.write(folder/'live.json',rec);rc=child.wait()
  rec.update(rc=rc,seconds=time.monotonic()-start,finished_at=r.stamp());r.write(folder/'results.json',[rec]);assert rc==0
  assert all(plan.sha(REPO/n)==h for n,h in sources.items())
  r.write(folder/'completed.json',dict(at=r.stamp(),CPU_only=True,GPU=False,benchmark_sha256=plan.sha(plan.BENCH),source_sha256=sources,revised_experimental_instance=6035,geometry_selector_ABI_unchanged=True,accumulator_banks=2))
  r.write(folder/'controller.json',dict(state='complete',pid=os.getpid(),at=r.stamp(),GPU=False))
  print('ACCUMULATOR_OFFICIAL_BUILD_PASS',rec['seconds'],plan.sha(plan.BENCH),flush=True)
 except BaseException as error:
  r.write(folder/'controller.json',dict(state='stopped',pid=os.getpid(),at=r.stamp(),GPU=False,error=str(error)));raise
if __name__=='__main__':main()
