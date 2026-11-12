"""P6l adapters around the preserved official-protocol command builders."""
from pathlib import Path
import importlib.util,os,json
ORIGINAL=Path('/workspace/composable_kernel/tools/grouped_gemm_depth8_reproduce/plan.py')
spec=importlib.util.spec_from_file_location('p6k_immutable_plan',ORIGINAL)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
REPO=base.REPO;BENCH=base.BENCH;BLAS=base.BLAS;SELECTOR=base.SELECTOR
PREFIX=base.PREFIX;V7=base.V7;DEPTH8=base.DEPTH8;sha=base.sha;digest=base.digest
FLAGS=dict(base.FLAGS)
for key in ['ENABLE_BF16_NN_DECODE_MAINLINE','DISABLE_BF16_NN_DECODE_MAINLINE','ENABLE_BF16_NN_DECODE_SHORT','DISABLE_BF16_NN_DECODE_SHORT']:
 FLAGS[PREFIX+key]={'0','1'}
FLAGS[PREFIX+'BF16_NN_DECODE_MAINLINE_VARIANT']={str(i) for i in range(21)}
FLAGS[PREFIX+'BF16_NN_DECODE_SHORT_VARIANT']={str(i) for i in range(6)}
def environment(overrides,inherited=None):
 assert all(k in FLAGS and v in FLAGS[k] for k,v in overrides.items()),overrides
 src=os.environ if inherited is None else inherited
 env={k:v for k,v in src.items() if not k.startswith(PREFIX) and k not in ['LD_PRELOAD','ROCR_VISIBLE_DEVICES','HSA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
 env.update(overrides);env.update(HIP_VISIBLE_DEVICES='4',CUDA_VISIBLE_DEVICES='4',OMP_NUM_THREADS='4')
 env['LD_LIBRARY_PATH']=str(BLAS.parent)+':'+env.get('LD_LIBRARY_PATH','')
 return env
for name in ['EARLY_B','XOR_A']:
 for mode in ['ENABLE','DISABLE']:
  FLAGS[PREFIX+mode+'_BF16_NN_DECODE_SHARED_'+name]={'0','1'}
command=base.command

for mode in ['ENABLE','DISABLE']:
 FLAGS[PREFIX+mode+'_BF16_NN_DECODE_FAST_POLICY']={'0','1'}
