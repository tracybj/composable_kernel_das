"""Two focused diagnostic profiles after accepted representative accuracy/timing."""
import json,uuid
from pathlib import Path
import runtime as r
import campaign as c
binary=r.sha(c.BENCH)
accepted=[json.loads(x) for x in (r.ROOT/'shape_results.jsonl').read_text().splitlines()]
assert any(x['suite']=='representatives' and x['id']==76 and x['binary_sha256']==binary for x in accepted)
for ident in [76,633,772]:
 assert any(x['suite']=='representatives' and x['id']==ident and x['binary_sha256']==binary for x in accepted)
 shape=c.shapes()[ident]
 with r.locked():
  for variant,filter in [('base','GemmPipelineV7Uniform' if ident==772 else 'GemmPipelineV7SmallMK32'),('new','DecodeNnWaveKernel')]:
   key='profile_'+str(ident)+'_'+variant+'_'+uuid.uuid4().hex[:10]
   spec=c.bench(shape,variant,'perf',key)
   tile_m=16 if shape['M']<=16 else 32 if shape['M']<=32 else 64
   spec.update(kind='profile',binary=str(c.BENCH),logical_tile=([256,256,16] if ident==772 else [64,128,32]) if variant=='base' else [tile_m,128,32])
   command=[x.replace('--samples=30','--samples=1').replace('--validate=1','--validate=0') for x in spec['command']]
   spec['command']=['/opt/rocm-6.3.3/bin/xprof','--kernels',filter,'--sections','speed_of_light,memory_workload,occupancy,memory_chart,instruction_statistics,compute_workload,scheduler_statistics,wave_state','--output-dir',str(r.ROOT/'runs'/key/'profile')]+command
   r.execute(spec)
print('FOCUSED_PROFILES_PASS')
