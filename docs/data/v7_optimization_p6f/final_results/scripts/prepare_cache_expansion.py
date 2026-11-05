"""Full12-case attribute domain; reuse the four already complete primary pairs."""
import csv,json
from pathlib import Path
import campaign as c
import runtime as r
domain=json.loads((r.ROOT/'cache_grid_domain_predeclared.json').read_text());ids=domain['ids'];assert len(ids)==12
best={int(x['ID']):x for x in csv.DictReader(open('/workspace/composable_kernel/docs/data/v7_best_known/best_current_1193.csv'))}
comparisons=json.loads((r.ROOT/'cache_comparisons.json').read_text());reuse=[x for x in comparisons if x['id'] in ids and x['variant']=='m5_n1']
assert len(reuse)==4
new_ids=[i for i in ids if i not in {x['id'] for x in reuse}]
cs=c.cases();jobs=[]
def candidate(i):
    walk=5 if cs[i]['N']//256>16 else 1
    return (5012 if walk==5 else 5005),dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK=str(walk),CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC='1')
for i in new_ids:
    num,e=candidate(i)
    for arm,expected,env in [('control',int(best[i]['best_instance_id']),json.loads(best[i]['enable_env'])),('candidate',num,e)]:
        spec=c.bench(f'cache_grid_acc_id{i}_{arm}',cs[i],expected,env,repeat=1);spec.update(task=2,arm=arm,variant='cache_grid');jobs.append(spec)
for rnd in [1,2,3]:
    for i in new_ids:
        num,e=candidate(i);pair=f'cache_grid_r{rnd}_id{i}'
        for arm,expected,env in [('control',int(best[i]['best_instance_id']),json.loads(best[i]['enable_env'])),('candidate',num,e)]:
            spec=c.bench(pair+'_'+arm,cs[i],expected,env,kind='perf');spec.update(task=2,arm=arm,round=rnd,pair=pair,variant='cache_grid',baseline_scheme=best[i]['best_scheme']);jobs.append(spec)
path=r.ROOT/'cache_grid_expansion_frozen.json';assert not path.exists()
path.write_text(json.dumps(dict(at=r.stamp(),domain=domain,new_ids=new_ids,reuse_primary_comparisons=reuse,candidate_rule='NoSLC, M01=5 if N/256>16 else1; fixed before expansion',jobs=jobs),indent=2))
print('CACHE_GRID_EXPANSION_FROZEN domain12 new8 accuracy16 perf48 jobs='+str(len(jobs)))
