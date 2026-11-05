"""CPU-only archive build alongside profiling; never relinks the live executable."""
import json,shutil,subprocess,time
import runtime as r

r.no_core();bench=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
before=r.sha(bench);backup=r.ROOT/'bench_cache'
assert before=='0491e577fc7da6c68f69aa726f285b09dd78d0e003720dba20584a4db21d5fe0'
assert not backup.exists();shutil.copy2(bench,backup)
cmd=['cmake','--build','build','--target','ck_tile_hcu_grouped_gemm_device_args_providers','ck_tile_hcu_grouped_gemm_v7_v8_bf16','-j4']
log=r.ROOT/'build_impl_static.log';assert not log.exists()
record=dict(at=r.stamp(),command=cmd,cpu_only=True,benchmark_sha_before=before,
            note='Only two static archive targets; no executable target and no HIP launches. Official executable remains frozen for in-flight serial profiler.')
(r.ROOT/'build_impl_static_command.json').write_text(json.dumps(record,indent=2))
start=time.monotonic()
with log.open('w') as f:
    result=subprocess.run(cmd,cwd=r.REPO,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,r.remaining()-900))
record.update(rc=result.returncode,seconds=time.monotonic()-start,finished_at=r.stamp(),benchmark_sha_after=r.sha(bench))
r.append(r.ROOT/'builds.jsonl',record)
assert before==record['benchmark_sha_after'],'CPU archive target unexpectedly changed executable'
print('IMPL_STATIC_BUILD',result.returncode,round(record['seconds'],2),flush=True)
raise SystemExit(result.returncode)
