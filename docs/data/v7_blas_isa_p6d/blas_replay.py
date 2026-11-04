#!/usr/bin/env python3
"""Replay exact frozen Native GroupedGemm algorithms, without retuning."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
TUNING=Path('/workspace/Grouped_Geem/blaslt_tuning_tools')
sys.path.insert(0,str(TUNING))
from tuning_common import Desc, Native, restore_inputs, sha

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--id',type=int,required=True)
 ap.add_argument('--warmup',type=int,default=10)
 ap.add_argument('--samples',type=int,default=30)
 ap.add_argument('--repeats',default='1,20')
 ap.add_argument('--validate',type=int,default=1)
 args=ap.parse_args()
 assert os.environ['HIP_VISIBLE_DEVICES']=='4' and os.environ['CUDA_VISIBLE_DEVICES']=='4'
 case=next(c for c in json.loads((ROOT/'cases.json').read_text()) if c['id']==args.id)
 cfgpath=Path('/workspace/v7_coverage/p6c/new_blas/run_config.json') if args.id>=1179 else TUNING/'results/bf16_nn_compare_20261027_191315/groupgemm/run_config.json'
 cfg=json.loads(cfgpath.read_text())
 cfg.update(gpu=4,binary=str(TUNING/'build/libgroupgemm_tune.so'))
 assert sha(cfg['blas_library'])==cfg['blas_sha256']
 import torch
 torch.set_num_threads(4)
 torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_device(0)
 g,m,n,k=(case[x] for x in ['G','M','N','K'])
 a=torch.empty((g,m,k),device='cuda',dtype=torch.bfloat16)
 b=torch.empty((g,k,n),device='cuda',dtype=torch.bfloat16)
 restore_inputs((a,b),tuple((cfg['seed']+args.id+i)%(1<<64) for i in range(2)))
 raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.bfloat16)
 c=raw[:,16:-16].view(g,m,n)
 c.fill_(float('nan'))
 descs=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr()) for i in range(g)])
 native=Native(cfg,case,descs)
 try:
  candidate=next(x for x in native.candidates if x['algorithm_id']==case['algorithm_id'])
  assert candidate['kernel_name']==case['kernel_name'],candidate
  print('REPLAY '+json.dumps(dict(case=case,candidate=candidate,gpu=4)),flush=True)
  results=[]
  for repeat in map(int,args.repeats.split(',')):
   times=[native.time(candidate['candidate_index'],descs,args.warmup if i==0 else 0,repeat) for i in range(args.samples)]
   results.append(dict(repeat=repeat,samples_ms=times,median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),cv=statistics.pstdev(times)/statistics.mean(times)))
  result=dict(id=args.id,gpu=4,backend='native_groupedgemm',algorithm_id=candidate['algorithm_id'],kernel_name=candidate['kernel_name'],timings=results)
  if args.validate:
   assert args.id!=1178,'Large prefill already validated in frozen baseline; avoid huge CPU GEMM'
   reference=torch.bmm(a.cpu().float(),b.cpu().float()).to(torch.bfloat16)
   actual=c.cpu();guards=raw.cpu()
   result.update(reference='cpu_fp32_bmm',checked=c.numel(),failed=int((actual!=reference).sum()),max_abs=float((actual.float()-reference.float()).abs().max()),guards_ok=bool((guards[:,:16]==117).all() and (guards[:,-16:]==117).all()))
   assert result['failed']==0 and result['max_abs']==0 and result['guards_ok'],result
  print('RESULT '+json.dumps(result),flush=True)
 finally:native.close()

if __name__=='__main__':main()
