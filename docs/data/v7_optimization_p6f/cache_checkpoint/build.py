"""Official CMake build, recorded once per tag, bounded by session deadline."""
import json,subprocess,sys,time
import runtime as r
tag=sys.argv[1]
assert tag.replace('_','').isalnum()
targets=['tile_example_grouped_gemm_'+x for x in ['device_args_benchmark','selector_test','device_args_query_test','device_args_smoke']]
cmd=['cmake','--build','build','--target',*targets,'-j4']
with r.locked():
    r.no_core();p=r.ROOT/('build_'+tag+'.log');assert not p.exists()
    record=dict(at=r.stamp(),command=cmd,tag=tag,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.REPO,text=True).strip())
    (r.ROOT/('build_'+tag+'_command.json')).write_text(json.dumps(record,indent=2))
    start=time.monotonic()
    with p.open('w') as f:proc=subprocess.run(cmd,cwd=r.REPO,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,r.remaining()-300))
    record.update(rc=proc.returncode,seconds=time.monotonic()-start,finished_at=r.stamp())
    r.append(r.ROOT/'builds.jsonl',record)
    print('BUILD_RESULT',tag,proc.returncode,round(record['seconds'],2),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)
