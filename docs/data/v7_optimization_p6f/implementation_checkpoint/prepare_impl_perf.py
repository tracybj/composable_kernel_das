"""Freeze finite tail/decode performance and necessary baseline precision controls."""
import csv,json,sys
import campaign as c
import runtime as r

p=r.ROOT;cs=c.cases()
best={int(x['ID']):x for x in csv.DictReader(open(r.REPO/'docs/data/v7_best_known/best_current_1193.csv'))}
def write(name,jobs,reason):
    target=p/name;assert not target.exists()
    target.write_text(json.dumps(dict(at=r.stamp(),reason=reason,jobs=jobs),indent=2))
def baseline(i):return int(best[i]['best_instance_id']),json.loads(best[i]['enable_env'])
def control_acc(i,prefix,task):
    num,env=baseline(i);x=c.bench(f'{prefix}_baseline_acc_id{i}',cs[i],num,env,repeat=3)
    x.update(task=task,arm='control',baseline_scheme=best[i]['best_scheme']);return x

original=json.loads((p/'tail_accuracy_frozen.json').read_text())['jobs']
tail=list(original)
for x in original:
    x=dict(x);x['key']=x['key'].replace('tail_','tailwide_',1)
    x['env']=dict(x['env'],CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL='2')
    if x.get('expected')==5013:x['expected']=5016
    if x.get('expected_not'):x['expected_not']=[5013,5016]
    tail.append(x)
tail.extend(control_acc(i,'tail',1) for i in range(139,155))
write('tail_accuracy_both_tiles_frozen.json',tail,'Tail128 and tail256 full16 and complete boundary/fallback checks; fresh current-route baseline precision')

tailperf=[]
for rnd in (1,2,3):
    for i in range(139,155):
        for mode,num,name in [(1,5013,'tail128'),(2,5016,'tail256')]:
            pair=f'{name}_r{rnd}_id{i}';bn,be=baseline(i)
            for arm,expected,env in [('control',bn,be),('candidate',num,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL=str(mode)))]:
                x=c.bench(pair+'_'+arm,cs[i],expected,env,kind='perf')
                x.update(task=1,round=rnd,pair=pair,arm=arm,variant=name,baseline_scheme=best[i]['best_scheme'])
                tailperf.append(x)
write('tail_perf_frozen.json',tailperf,'All16 original N-tail shapes, both independent geometries vs fresh current best; no winner selection')

ids=[76,633,681,620,621,358,772,622,688]+list(range(1179,1194))
assert len(ids)==len(set(ids))==24
decode=json.loads((p/'decode_accuracy_frozen.json').read_text())['jobs']
decode+= [control_acc(i,'steady',3) for i in ids]
write('decode_accuracy_with_baselines_frozen.json',decode,'Original phase/lifetime stress and new15 plus fresh per-shape current-best baseline precision')
perf=[]
for rnd in (1,2,3):
    for i in ids:
        for mode,num,name in [(5,5014,'k32_steady'),(6,5015,'k16_steady')]:
            pair=f'{name}_r{rnd}_id{i}';bn,be=baseline(i)
            cn=num if cs[i]['M']<=64 else 5000
            for arm,expected,env in [('control',bn,be),('candidate',cn,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=str(mode)))]:
                x=c.bench(pair+'_'+arm,cs[i],expected,env,kind='perf')
                x.update(task=3,round=rnd,pair=pair,arm=arm,variant=name,baseline_scheme=best[i]['best_scheme'],
                         fallback=cs[i]['M']>64)
                perf.append(x)
write('decode_representative_perf_frozen.json',perf,'24 declared representatives including all15new and M128/256 fallback controls; current-known best per shape, not obsolete5000')

acc=[];perf=[]
for i in [620,621,76,358]:
    for arm,num,env in [('control',5003,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='128')),
                        ('candidate',5017,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='7'))]:
        x=c.bench(f'small_cache_acc_id{i}_{arm}',cs[i],num,env,repeat=3);x.update(task=2,arm=arm,variant='small_cache');acc.append(x)
    for rnd in (1,2,3):
        pair=f'small_cache_r{rnd}_id{i}'
        for arm,num,env in [('control',5003,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='128')),
                            ('candidate',5017,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='7'))]:
            x=c.bench(pair+'_'+arm,cs[i],num,env,kind='perf');x.update(task=2,round=rnd,pair=pair,arm=arm,variant='small_cache',
                note='Matched5003 geometry/cache-bit isolation; current-best comparison remains separate for ID76')
            perf.append(x)
write('small_cache_accuracy_frozen.json',acc,'Only four controls permitted; no cache/tile Cartesian expansion')
write('small_cache_perf_frozen.json',perf,'Four matched5003 SLC vs5017 NoSLC controls; no general adoption without complete domain data')

native=[]
for rnd in (1,2,3):
    for i in [146,147,76,620,621,1186,1187]:
        script=p/'native_replay.py'
        native.append(dict(key=f'impl_native_r{rnd}_id{i}',id=i,round=rnd,kind='native_perf',replay_script_sha256=r.sha(script),
            command=[sys.executable,'-u',str(script),f'--id={i}','--warmup=10','--samples=30','--repeats=1,20','--validate=0'],env=dict(LD_PRELOAD=r.BLAS)))
write('impl_native_frozen.json',native,'Frozen algorithms; same-card single-launch and repeat20 diagnostic, never replacing historical R')
print('IMPL_PLANS_FROZEN tail_accuracy',len(tail),'tail_perf',len(tailperf),'decode_accuracy',len(decode),'decode_perf',288,'small_control',len(acc)+len(perf),'native',len(native))
