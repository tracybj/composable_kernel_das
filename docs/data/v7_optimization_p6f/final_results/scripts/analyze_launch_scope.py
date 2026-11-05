"""Separate event-batch diagnostics, retaining all CVs and same-epoch complete pairs."""
import json,statistics
import runtime as r

raw=[json.loads(s) for s in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
groups={}
for x in raw:
    if x.get('ok') and x.get('kind')=='diagnostic_perf':
        groups.setdefault((x['id'],x['variant'],x['epoch'],x['binary_sha256']),{})[(x['round'],x['arm'])]=x
rows=[]
native=json.loads((r.ROOT/'native_diagnostic_summary.json').read_text())
for (i,variant,epoch,binary),data in sorted(groups.items()):
    if not all((rd,a) in data for rd in (1,2,3) for a in ['control','candidate']):continue
    before=[data[(rd,'control')] for rd in (1,2,3)];after=[data[(rd,'candidate')] for rd in (1,2,3)]
    assert {x['launches_per_sample'] for x in before}=={1} and {x['launches_per_sample'] for x in after}=={20}
    assert len({x['selected'] for x in before+after})==1
    med=statistics.median;b=[x['result']['median_ms'] for x in before];a=[x['result']['median_ms'] for x in after]
    refs=[x for x in native if x['id']==i and x['epoch']==epoch and x['study']=='launch_scope_native']
    assert len(refs)==1,(i,epoch)
    rows.append(dict(id=i,variant=variant,epoch=epoch,physical_gpu=4,instance_id=before[0]['selected'],
        CK_single_ms=med(b),CK_batch20_per_launch_ms=med(a),single_round_ms=b,batch20_round_per_launch_ms=a,
        single_cv=[x['result']['cv'] for x in before],batch20_cv=[x['result']['cv'] for x in after],
        amortized_difference_us=1000*(med(b)-med(a)),Native_same_epoch=refs[0],binary_sha256=binary,
        source_keys=[x['key'] for x in before+after],
        note='Diagnostic scope only;20-launch averages are excluded from formal one-launch/frozen-R/envelope views. Public API/default selector invoked each launch.'))
(r.ROOT/'launch_scope_results.json').write_text(json.dumps(rows,indent=2))
for x in rows:print('LAUNCH_SCOPE_DIAG',x['id'],x['instance_id'],round(x['CK_single_ms'],6),'->',round(x['CK_batch20_per_launch_ms'],6),'difference_us',round(x['amortized_difference_us'],3))
print('DIAGNOSTIC_COMPLETE',len(rows),'expected6')
