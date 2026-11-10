"""Apply the predeclared confirmation and uncertainty rules without winner lists."""
import json,statistics
from pathlib import Path
import runtime as r
import plan

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'summary.json').read_text())
assert s['discovery_complete'] and not s['unresolved_failed_jobs'] and not s['unresolved_incomplete_raw_keys']
confirm=json.loads((ROOT/'confirmation_plan.json').read_text())
rows=json.loads((ROOT/'results.json').read_text())
confirmation=[x for x in rows if x['suite']=='confirmation']
assert len(confirmation)==len(confirm['cases'])
decisions={}
for name,discovery in s['regions'].items():
    if name not in confirm['predicates']:
        decisions[name]=dict(status='not_qualified_in_discovery',predicate=discovery['predicate'],
            reason='Did not pass whole-predicate discovery criterion or had fewer than four unseen probes',
            discovery=discovery['combined'])
        continue
    rs=[x for x in confirmation if plan.region(x,name)]
    before=sum(x['before_ms'] for x in rs);after=sum(x['after_ms'] for x in rs)
    stats=dict(cases=len(rs),improved_gt2=sum(x['change_pct']<-2 for x in rs),
        regressed_gt2=sum(x['change_pct']>2 for x in rs),
        three_round_improved_gt2=sum(x['three_round_improved_gt2'] for x in rs),
        high_cv=sum(x['high_cv'] for x in rs),median_change_pct=statistics.median(x['change_pct'] for x in rs),
        worst_change_pct=max(x['change_pct'] for x in rs),sum_change_pct=(after/before-1)*100)
    if stats['regressed_gt2'] or stats['median_change_pct']>-3:
        status='not_confirmed'
    elif stats['high_cv'] or discovery['combined']['high_cv']:
        status='direction_confirmed_but_sample_variation_unresolved'
    else:
        status='verified_bounded_scope'
    decisions[name]=dict(status=status,predicate=discovery['predicate'],discovery=discovery['combined'],confirmation=stats)
    if name=='n256_k512_g64':
        assert {x['M'] for x in rs}==set(range(1,33))
        decisions[name]['all_integer_m_1_through_32_checked']=True
result=dict(at=r.stamp(),complete=True,decisions=decisions,
    verified_scopes=[name for name,x in decisions.items() if x['status']=='verified_bounded_scope'],
    production_enabled=False,historical1193_updated=False,p6i_state='paused',p6g_state='paused',
    implementation_contract='Archived Depth8 on gfx936, packed dense BF16 NN, common nonempty/nonragged, N%128=0 and K%128=0; tested finite dimensions only',
    scope='Establish evidence for a property-based range; do not select winner shape IDs or silently remove high-CV/negative results')
(ROOT/'decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print('SCOPE_DECISION_PASS',json.dumps({name:x['status'] for name,x in decisions.items()}),flush=True)
