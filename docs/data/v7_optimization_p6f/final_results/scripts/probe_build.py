import json,subprocess,time,sys
import runtime as r
names=sys.argv[1:] or ['tail','k32steady','k16steady']
for spec in json.loads((r.ROOT/'probe_commands.json').read_text()):
    if spec['name'] not in names:continue
    path=r.ROOT/('probe_'+spec['name']+'_build.log');assert not path.exists()
    start=time.monotonic()
    with path.open('w') as f:p=subprocess.run(spec['command'],cwd=r.REPO,stdout=f,stderr=subprocess.STDOUT,timeout=min(1800,max(1,r.remaining()-300)))
    r.append(r.ROOT/'probe_builds.jsonl',dict(spec,at=r.stamp(),rc=p.returncode,seconds=time.monotonic()-start))
    print('PROBE_BUILD',spec['name'],p.returncode,round(time.monotonic()-start,2),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
