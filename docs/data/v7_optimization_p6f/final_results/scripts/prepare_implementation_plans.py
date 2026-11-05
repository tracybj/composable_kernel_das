"""Freeze accuracy/representative plans before implementation GPU results."""
import csv,json
from pathlib import Path
import campaign as c
import runtime as r
cs=c.cases()
tail_env=dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL='1')
def shape(m,n,k,g):return dict(id=-1,M=m,N=n,K=k,G=g)
tail=[]
for i in [139,143,145,146,147,151,153,154]+[x for x in range(139,155) if x not in [139,143,145,146,147,151,153,154]]:
    tail.append(c.bench(f'tail_acc_id{i}',cs[i],5013,tail_env,repeat=3))
for tag,s in [('n136',shape(1,136,64,4)),('n248',shape(7,248,80,4)),('n264',shape(63,264,208,17)),('n1240',shape(127,1240,1536,4)),('m128',shape(128,1232,64,4)),('g1024',shape(1,136,64,1024))]:
    tail.append(c.bench('tail_acc_'+tag,s,5013,tail_env,repeat=3))
for tag,s,extra,overrides in [
    ('n1233',shape(1,1233,1536,4),[],{}),('n1280',shape(1,1280,1536,4),[],{}),
    ('n120',shape(1,120,64,4),[],{}),('m129',shape(129,1232,1536,4),[],{}),
    ('k65',shape(1,1232,65,4),[],{}),('k4112',shape(1,1232,4112,4),[],{}),
    ('ragged',shape(1,1232,1536,4),['--group-lengths=1,65,7,129'],{}),
    ('empty',shape(1,1232,1536,4),['--group-lengths=1,65,0,7','--allow-empty-groups=1'],{}),
    ('disable',shape(1,1232,1536,4),[],dict(CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_N_TAIL='1')),
    ('global_off',shape(1,1232,1536,4),[],dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN='0'))]:
    j=c.bench('tail_fallback_'+tag,s,None,dict(tail_env,**overrides),repeat=3);j['command']+=extra;j['expected_not']=[5013];tail.append(j)
for j in tail:j['task']=1
(r.ROOT/'tail_accuracy_frozen.json').write_text(json.dumps(dict(at=r.stamp(),jobs=tail,reason='full16 plus vector, M/K/G/ragged/empty/disable contracts'),indent=2))
decode=[]
for name,mode,num,stageK in [('k32',5,5014,32),('k16',6,5015,16)]:
    e=dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=str(mode))
    for S in list(range(4,29))+[48,64,128]:
        decode.append(c.bench(f'steady_acc_{name}_stages{S}',shape(7,256,S*stageK,4),num,e,repeat=3))
    for m in [1,31,63,64]:decode.append(c.bench(f'steady_acc_{name}_m{m}',shape(m,384,512,17),num,e,repeat=3))
    for g in [1,16,1024]:decode.append(c.bench(f'steady_acc_{name}_g{g}',shape(1,128,128 if name=='k32' else 64,g),num,e,repeat=3))
    for i in [76,633,681,620,621,358,772]:decode.append(c.bench(f'steady_acc_{name}_id{i}',cs[i],num if cs[i]['M']<=64 else None,e,repeat=3))
    for i in range(1179,1194):
        decode.append(c.bench(f'steady_acc_{name}_newid{i}',cs[i],num if cs[i]['M']<=64 else 5000,e,repeat=3))
    for m in [65,128,256]:
        j=c.bench(f'steady_fallback_{name}_m{m}',shape(m,256,512,4),5000,e,repeat=3);decode.append(j)
    for tag,s,extra in [('n1232',shape(1,1232,1536,4),[]),('ragged',shape(1,256,512,4),['--group-lengths=1,65,7,129']),('empty',shape(1,256,512,4),['--group-lengths=1,65,0,7','--allow-empty-groups=1'])]:
        j=c.bench(f'steady_fallback_{name}_{tag}',s,None,e,repeat=3);j['command']+=extra;j['expected_not']=[5014,5015];decode.append(j)
    for S in [4,5,16,17]:decode.append(c.bench(f'steady_stress_{name}_stages{S}',shape(1,128,S*stageK,32),num,e,repeat=20))
for j in decode:j['task']=3
(r.ROOT/'decode_accuracy_frozen.json').write_text(json.dumps(dict(at=r.stamp(),jobs=decode,reason='all short prologue/tail/phase residues and full-cycle boundaries, new15, fallback, repetition stress'),indent=2))
print('ACCURACY_PLANS_FROZEN tail='+str(len(tail))+' decode='+str(len(decode)))
