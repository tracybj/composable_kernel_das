"""Freeze a compact profiler set; exact kernel filters include the new bodies."""
import json
import campaign as c
import runtime as r

cs=c.cases();tail=[];decode=[]
def profile(target,i,name,num,env,tile,filter):
    x=c.bench(f'impl_profile_{name}_id{i}',cs[i],num,env,kind='profile')
    x['command']=[a.replace('--warmup=1','--warmup=10') for a in x['command']]
    x.update(task=1 if target is tail else 3,variant=name,logical_tile=tile,kernel_filter=filter)
    target.append(x)
for i in [146,147]:
    profile(tail,i,'tail_control',4003,{},[128,128,32],'GroupedGemmKernel')
    profile(tail,i,'tail128',5013,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL='1'),[128,128,16],'GemmPipelineV7Tail128')
    profile(tail,i,'tail256',5016,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL='2'),[128,256,16],'GemmPipelineV7Tail256')
for i in [76,620,621]:
    profile(decode,i,'small_k16',5003,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M='128'),[64,128,16],'GemmPipelineV7SmallM<')
    profile(decode,i,'k16_steady',5015,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='6'),[64,128,16],'GemmPipelineV7SmallMK16Steady')
for mode,num,name,filter,k in [(1,5007,'old_k16_drain','GemmPipelineV7SmallMDrain',16),
                              (2,5008,'old_k32','GemmPipelineV7SmallMK32<',32),
                              (5,5014,'k32_steady','GemmPipelineV7SmallMK32Steady',32)]:
    profile(decode,76,name,num,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=str(mode)),[64,128,k],filter)
for i in [620,621]:
    profile(decode,i,'small_noslc',5017,dict(CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT='7'),[64,128,16],'GemmPipelineV7SmallMNoSlc')
for name,jobs in [('tail_profiles_frozen.json',tail),('decode_profiles_frozen.json',decode)]:
    path=r.ROOT/name;assert not path.exists();path.write_text(json.dumps(dict(at=r.stamp(),jobs=jobs,
        note='Exact filters must be checked against actual final binary symbol table before running; control4003 geometry must be filled from actual symbol/dispatch'),indent=2))
print('IMPL_PROFILE_PLANS_FROZEN',len(tail),len(decode))
