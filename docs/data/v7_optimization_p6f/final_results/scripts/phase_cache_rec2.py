import subprocess,sys,time
from pathlib import Path
p=Path('/workspace/v7_coverage/p6f')
for suite,plan in [('cache_grid_expansion_rec2','cache_grid_expansion_rec2_frozen.json'),
                   ('cache_profiles_rec2','cache_profiles_rec2_frozen.json')]:
    output=p/(suite+'_controller.log')
    assert not output.exists()
    with output.open('w') as log:
        result=subprocess.run([sys.executable,'-u',str(p/'campaign.py'),suite,'--plan',str(p/plan)],stdout=log,stderr=subprocess.STDOUT)
    print(suite,result.returncode,flush=True)
    if result.returncode:raise SystemExit(result.returncode)
