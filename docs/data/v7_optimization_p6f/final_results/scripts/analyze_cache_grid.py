"""Combine the frozen twelve-case policy; reuse only explicitly declared primary pairs."""
import json,statistics
import runtime as r

p=r.ROOT
plan=json.loads((p/'cache_grid_expansion_rec2_frozen.json').read_text())
allrows=json.loads((p/'per_shape_results.json').read_text())
rows=[dict(x,reused_primary=False) for x in allrows if x['variant']=='cache_grid']
for old in plan['reuse_primary_comparisons']:
    x=next(x for x in allrows if x['id']==old['id'] and x['variant']=='m5_n1')
    rows.append(dict(x,variant='cache_grid',reused_primary=True))
ids=plan['domain']['ids'];assert len(rows)==12 and sorted(x['id'] for x in rows)==sorted(ids)
assert all(x['candidate_instance']==(5012 if x['N']>4096 else 5005) for x in rows)
reg=[x for x in rows if x['paired_delta_pct']>2]
changed=[x for x in rows if x['candidate_instance']!=x['control_instance']]
summary=dict(at=r.stamp(),domain=plan['domain'],count=12,
    median_paired_change_pct=statistics.median(x['paired_delta_pct'] for x in rows),
    equal_call_paired_total_change_pct=statistics.median(100*(sum(x['after_round_ms'][rd] for x in rows)/sum(x['before_round_ms'][rd] for x in rows)-1) for rd in range(3)),
    regressions_over2pct=[x['id'] for x in reg],
    uncertain=[x['id'] for x in rows if x['quality']!='stable_direction'],
    same_instance_controls=[x['id'] for x in rows if x['candidate_instance']==x['control_instance']],
    recommendation='candidate for bounded opt-in policy verification' if not reg and all(x['paired_delta_pct']< -2 for x in changed) else 'do not adopt as entire domain without further evidence',
    note='Whole contrast same-epoch/same-binary/same-GPU; different completed shapes may have different epochs; full domain fixed before expansion; not production frequency weighted')
(p/'cache_grid_results.json').write_text(json.dumps(sorted(rows,key=lambda x:x['id']),indent=2))
(p/'cache_grid_summary.json').write_text(json.dumps(summary,indent=2))
print('CACHE_GRID_COMPLETE',summary['count'],'median',round(summary['median_paired_change_pct'],3),'total',round(summary['equal_call_paired_total_change_pct'],3),'regressions',summary['regressions_over2pct'],'uncertain',summary['uncertain'])
for x in sorted(rows,key=lambda x:x['id']):print(x['id'],x['control_instance'],x['candidate_instance'],round(x['before_ms'],6),round(x['after_ms'],6),round(x['paired_delta_pct'],3),x['quality'])
