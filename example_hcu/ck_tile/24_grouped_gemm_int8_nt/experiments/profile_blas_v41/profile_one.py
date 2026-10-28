"""Replay frozen BLASLt algorithm6950; no tuning and no baseline overwrite."""
from pathlib import Path
import sys,os,ctypes as C,json,hashlib,statistics,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
assert os.environ.get('HIP_VISIBLE_DEVICES')=='7' and os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
base=Path('/workspace/GroupGemm/baseline/blaslt_tuning_tools');sys.path.insert(0,str(base))
from tuning_common import Native,Desc,restore_inputs,inputs_ok
run=base/'results/int8_nt_dual_20261027_191653/groupgemm'
cfg=json.loads((run/'run_config.json').read_text());cfg['binary']=str(run/'bin/libgroupgemm_tune.so')
assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==cfg['binary_sha256']
case=next(x for x in cfg['cases'] if x['id']==1177)
g,m,n,k=[case[x] for x in ['G','M','N','K']]
a=torch.empty((g,m,k),device='cuda',dtype=torch.int8);b=torch.empty((g,n,k),device='cuda',dtype=torch.int8)
seeds=(2026102801,2026102802);restore_inputs((a,b),seeds)
raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n);c.fill_(123456789)
ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr()) for i in range(g)])
native=Native(cfg,case,ds)
try:
 selected=[x for x in native.candidates if x['algorithm_id']==6950]
 assert len(selected)==1,selected
 selected=selected[0];index=selected['candidate_index']
 assert 'GG_AI_SAV_MT256x256x32' in selected['kernel_name'],selected
 torch.cuda.synchronize()
 mode=sys.argv[1] if len(sys.argv)>1 else 'profile'
 times=[native.time(index,ds,0,1)] if mode=='profile' else [native.time(index,ds,5,20) for _ in range(3)]
 bad=0
 for i in range(g):bad+=int((c[i]!=(a[i].float()@b[i].float().T).to(torch.int32)).sum())
 guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
 unchanged=inputs_ok((a,b),seeds)
 assert bad==0 and guard and unchanged,(bad,guard,unchanged)
 result=dict(M=m,N=n,K=k,G=g,gpu=7,backend='groupgemm',selected=selected,mode=mode,samples_ms=times,median_ms=statistics.median(times),TOPS=2*g*m*n*k/statistics.median(times)/1e9,bad_elements=bad,output_guards=guard,full_inputs_unchanged=unchanged)
 print(json.dumps(result),flush=True)
 E=Path(__file__).resolve().parents[2];(E/f'results/profile_v41_blas/{mode}_result.json').write_text(json.dumps(result,indent=2))
finally:native.close()
