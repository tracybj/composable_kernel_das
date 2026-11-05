"""Freeze a whole resource domain, before its first GPU expansion result."""
import csv,json
import campaign as c
import runtime as r

cs=c.cases();best={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv'))}
def eligible(x):
    ct=((x['M']+63)//64)*(x['N']//128)*x['G']
    return x['M']<=64 and 2048<=x['K']<=4096 and x['K']%32==0 and x['N']%128==0 and x['G']<=1024 and 80<=ct<=640
ids=[i for i,x in cs.items() if eligible(x)];assert len(ids)==159
domain=dict(at=r.stamp(),count=len(ids),ids=ids,
    predicate='common nonragged/nonempty BF16 NN gfx936;M<=64,N%128=0,2048<=K<=4096,K%32=0,G<=1024,80<=ceil(M/64)*ceil(N/128)*G<=640; per-group byte/tile limits',
    resource_basis='PhysicalGPU4 has80CUs; at least one useful CTA/CU, at most eight grid rounds; longK amortizes fixed tail cost. Entire appendix domain, including N6144/7168 near-neighbors, not a winner list.',
    reason='620/621 K32 vsbest5003 improves11/14%; matched5008 vs5014 within2%, so test existing5008 geometry instead of crediting unproved refactor gain. NewlowCTA21 and K16short189 have no representative benefit supporting their expansion.',
    acceptance='No global adoption if any stable regression>2%; keep all negatives/uncertain rows. Any later narrowed rule must be attributes, frozen and freshly verified, not IDs.')
q=r.ROOT/'k32_resource_domain_predeclared.json';assert not q.exists();q.write_text(json.dumps(domain,indent=2))
jobs=[]
for i in ids:
    bn=int(best[i]['best_instance_id']);be=json.loads(best[i]['enable_env'])
    for arm,num,env in [('control',bn,be),('candidate',5008,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='2'))]:
        x=c.bench(f'k32_resource_acc_id{i}_{arm}',cs[i],num,env,repeat=1);x.update(task=3,arm=arm,variant='k32_resource');jobs.append(x)
for i in ids:
    bn=int(best[i]['best_instance_id']);be=json.loads(best[i]['enable_env'])
    for rnd in (1,2,3):
        pair=f'k32_resource_r{rnd}_id{i}'
        for arm,num,env in [('control',bn,be),('candidate',5008,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='2'))]:
            x=c.bench(pair+'_'+arm,cs[i],num,env,kind='perf');x.update(task=3,arm=arm,round=rnd,pair=pair,variant='k32_resource',baseline_scheme=best[i]['best_scheme']);jobs.append(x)
q=r.ROOT/'k32_resource_domain_frozen.json';assert not q.exists();q.write_text(json.dumps(dict(at=r.stamp(),domain=domain,jobs=jobs),indent=2))
print('K32_RESOURCE_DOMAIN_FROZEN shapes159 accuracy318 perf954')
