"""Freeze small launch-scope diagnostics; no values enter formal one-launch views."""
import json,sys
import campaign as c
import runtime as r

cs=c.cases();configs=[(76,'old_k32',2,5008),(76,'steady_k32',5,5014),
                     (620,'current_k16',0,5003),(620,'steady_k32',5,5014),
                     (621,'current_k16',0,5003),(621,'steady_k32',5,5014)]
accuracy=[];perf=[];native=[]
for i,name,mode,num in configs:
    env=(dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='128') if mode==0 else dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=str(mode)))
    x=c.bench(f'launch_scope_acc_{name}_id{i}',cs[i],num,env,repeat=3);x['task']=3;accuracy.append(x)
    for rnd in (1,2,3):
        pair=f'launch_scope_{name}_r{rnd}_id{i}'
        for arm,launches in [('control',1),('candidate',20)]:
            x=c.bench(pair+'_'+arm,cs[i],num,env,kind='perf');x.update(kind='diagnostic_perf',task=3,arm=arm,round=rnd,pair=pair,variant='launch_scope_'+name,launches_per_sample=launches,
                protocol='diagnostic event batch normalized per launch; excluded from formal/frozen-R/envelope tables')
            # Control exercises the unchanged default; only the candidate sets20.
            if launches!=1:x['command'].append('--launches-per-sample='+str(launches))
            perf.append(x)
for i in [76,620,621]:
    for rnd in (1,2,3):
        script=r.ROOT/'native_replay.py'
        native.append(dict(key=f'launch_scope_native_r{rnd}_id{i}',id=i,round=rnd,kind='native_perf',replay_script_sha256=r.sha(script),command=[sys.executable,'-u',str(script),f'--id={i}','--warmup=10','--samples=30','--repeats=1,20','--validate=0'],env=dict(LD_PRELOAD=r.BLAS)))
for name,jobs in [('launch_scope_accuracy_frozen.json',accuracy),('launch_scope_perf_frozen.json',perf),('launch_scope_native_frozen.json',native)]:
    q=r.ROOT/name;assert not q.exists();q.write_text(json.dumps(dict(at=r.stamp(),reason='Native76 one-launch vsrepeat20 has6.75us per-launch difference; isolate measurement scope via explicit new default-off CK batch option, same public API/selector per launch, no historical reference replacement',jobs=jobs),indent=2))
print('LAUNCH_SCOPE_FROZEN accuracy6 native9 diagnostic36')
