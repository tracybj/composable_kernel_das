"""Freeze an independent confirmation set from whole qualifying predicates."""
import json,shutil
from pathlib import Path
import runtime as r
import plan

ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/'summary.json').read_text())
assert summary['discovery_complete']
assert not summary['unresolved_failed_jobs'] and not summary['unresolved_incomplete_raw_keys']
frozen=json.loads((ROOT/'scope_plan.json').read_text())
names=[name for name,x in summary['regions'].items() if x['eligible_for_confirmation']]
cases=[x for x in frozen['existing']+frozen['unseen_probes'] if any(plan.region(x,name) for name in names)]
if 'n256_k512_g64' in names:
    extension=json.loads((ROOT/'confirmation_extension.json').read_text())
    known={tuple(x[k] for k in ['M','N','K','G']) for x in cases}
    for x in extension['cases']:
        if tuple(x[k] for k in ['M','N','K','G']) not in known:
            cases.append(x)
result=dict(at=r.stamp(),predicates=names,cases=cases,
  selection='Every case satisfying any whole eligible predeclared predicate; no winner-ID selection',
  discovery_summary_sha256=r.sha(ROOT/'summary.json'),scope_plan_sha256=r.sha(ROOT/'scope_plan.json'),
  second_pass_rounds=3,accuracy_first=True)
p=ROOT/'confirmation_plan.json'
assert not p.exists()
for name in ['summary.json','results.json','results.csv']:
    shutil.copyfile(ROOT/name,ROOT/('discovery_'+name))
p.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('CONFIRMATION_PLAN_FROZEN predicates='+str(names)+' cases='+str(len(cases)),flush=True)
