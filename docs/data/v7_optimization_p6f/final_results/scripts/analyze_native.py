"""Fresh frozen-algorithm Native diagnostics; never overwrite historical references."""
import csv,json,statistics
import runtime as r

frozen={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv'))}
raw=[json.loads(s) for s in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
groups={}
for x in raw:
    if x.get('ok') and x.get('kind')=='native_perf':
        # Complete same-epoch study only; do not assemble rounds across recovery.
        study=x['logical_key'].split('_r',1)[0]
        groups.setdefault((study,x['id'],x['epoch']),{})[x['round']]=x
rows=[]
for (study,i,epoch),data in sorted(groups.items()):
    if set(data)!={1,2,3}:continue
    jobs=[data[rd] for rd in (1,2,3)]
    assert len({x['result']['algorithm_id'] for x in jobs})==1
    assert {x['physical_gpu'] for x in jobs}=={4}
    assert len({x['replay_script_sha256'] for x in jobs})==1
    ref=frozen[i];timings=[]
    for repeat in (1,20):
        values=[next(t for t in x['result']['timings'] if t['repeat']==repeat) for x in jobs]
        assert all(len(t['samples_ms'])==30 for t in values)
        med=statistics.median(t['median_ms'] for t in values)
        timings.append(dict(repeat=repeat,median_ms=med,round_ms=[t['median_ms'] for t in values],
            round_cv=[t['cv'] for t in values],versus_frozen_pct=100*(med/float(ref['BLAS_ms'])-1)))
    rows.append(dict(study=study,id=i,epoch=epoch,physical_gpu=4,algorithm_id=jobs[0]['result']['algorithm_id'],
        timings=timings,frozen_ms=float(ref['BLAS_ms']),frozen_gpu=int(ref['BLAS_gpu']),
        frozen_protocol=ref['BLAS_protocol'],source_keys=[x['key'] for x in jobs],
        note='Frozen algorithms, no retuning; same-card single-launch diagnostic is distinct from historical repeat20 R'))
(r.ROOT/'native_diagnostic_summary.json').write_text(json.dumps(rows,indent=2))
for x in rows:print('NATIVE',x['id'],'epoch',x['epoch'],'single/repeat20',[round(t['median_ms'],6) for t in x['timings']])
