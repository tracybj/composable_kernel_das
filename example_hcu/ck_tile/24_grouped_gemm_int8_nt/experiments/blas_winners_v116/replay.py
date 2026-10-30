"""Replay frozen per-shape Max TOPS winner,never retune or alter targets."""
from pathlib import Path
import json,csv,ctypes as C,sys,os,hashlib,resource,gc,statistics,argparse
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
assert all(os.environ.get(x)=='7' for x in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','DTK_VISIBLE_DEVICES'])
ap=argparse.ArgumentParser();ap.add_argument('--id',type=int);ap.add_argument('--profile',action='store_true');args=ap.parse_args()
e=Path(__file__).resolve().parents[2];out=e/'results/blas_winners_v116';out.mkdir(exist_ok=True)
frozen=json.loads((Path(__file__).parent/'winners.json').read_text());cases=[x for x in frozen['results'] if args.id is None or x['ID']==args.id];assert cases
base=Path('/workspace/GroupGemm/baseline/blaslt_tuning_tools');sys.path.insert(0,str(base));from tuning_common import Native,Desc
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc as CKDesc
hip=C.CDLL('/opt/dtk/lib/libgalaxyhip.so.5');pci=C.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(pci,64,0)==0;assert pci.value.decode().lower()=='0000:f5:00.0'
ckpath=e/'build/v114/libprobe.so';cksha=hashlib.sha256(ckpath.read_bytes()).hexdigest();assert cksha=='6ef4e295c783ccddc58f63d8657f6a6eaaff85dc6744f863a8e12a39a244376f'
ck=C.CDLL(str(ckpath));ck.probe_create.argtypes=[C.POINTER(CKDesc),C.c_int];ck.probe_create.restype=C.c_void_p;ck.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];ck.probe_run.restype=C.c_float;ck.probe_destroy.argtypes=[C.c_void_p]
allrows=[]
for case in cases:
 i=case['ID'];g,m,n,k=[case[x] for x in ['G','M','N','K']];run=base/'results/int8_nt_dual_20261027_191653'/case['winner'];cfg=json.loads((run/'run_config.json').read_text());cfg['binary']=str(run/'bin'/('lib'+case['winner']+'_tune.so'))
 assert hashlib.sha256(Path(cfg['binary']).read_bytes()).hexdigest()==cfg['binary_sha256'];assert hashlib.sha256(Path(cfg['blas_library']).read_bytes()).hexdigest()==cfg['blas_sha256'];assert cfg['precision']=='int8' and cfg['layout']=='NT'
 torch.manual_seed(20261030+i);a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8);olda=a.clone();oldb=b.clone();raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
 pb=b.reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous() if case['CK_route']>=5 else None;oldpb=pb.clone() if pb is not None else None
 desc=(Desc*g)(*[Desc(a[j].data_ptr(),b[j].data_ptr(),c[j].data_ptr()) for j in range(g)])
 native=Native(cfg,dict(id=i,G=g,M=m,N=n,K=k),desc);selected=[x for x in native.candidates if x['algorithm_id']==case['algorithm_id']];assert len(selected)==1,(i,selected);selected=selected[0];assert selected['kernel_name']==case['kernel_name'],(i,selected);index=selected['candidate_index']
 vv=pb if pb is not None else b;cd=(CKDesc*g)(*[CKDesc(a[j].data_ptr(),vv[j].data_ptr(),c[j].data_ptr(),1,m,n,k,k,k,n,0,None,None) for j in range(g)]);ctx=ck.probe_create(cd,g);assert ctx
 samples={'BLAS':[],'CK':[]};warm,repeat,samplecount=(2,1,1) if args.profile else (10,50,3)
 checks=[]
 try:
  for sample in range(samplecount):
   for backend in (['BLAS','CK'] if sample%2==0 else ['CK','BLAS']):
    c.fill_(123456789);torch.cuda.synchronize()
    ms=native.time(index,desc,warm,repeat) if backend=='BLAS' else float(ck.probe_run(ctx,50000+case['CK_route'],warm,repeat));assert ms>0;samples[backend].append(ms)
    bad=0
    for j in range(g):
     af,bf=a[j].float(),b[j].float();ref=(af@bf.T).to(torch.int32)
     for r0 in range(0,m,256):bad+=int((c[j,r0:r0+256]!=ref[r0:r0+256]).sum())
     del af,bf,ref
    guards=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=torch.equal(a,olda) and torch.equal(b,oldb) and (pb is None or torch.equal(pb,oldpb));assert bad==0 and guards and unchanged,(i,backend,bad,guards,unchanged);checks.append(dict(sample=sample,backend=backend,full_output=True,guards=True,full_inputs_unchanged=True))
  row=dict(ID=i,M=m,N=n,K=k,G=g,winner=case['winner'],algorithm_id=case['algorithm_id'],selected=selected,target_TOPS=case['target_TOPS'],times_ms=samples,throughput={name:2*g*m*n*k/statistics.median(v)/1e9 for name,v in samples.items()},checks=checks,profile=args.profile,physical_gpu=7,pci=pci.value.decode(),ck_module_sha256=cksha,blas_sha256=cfg['blas_sha256'],bridge_sha256=cfg['binary_sha256'],scope='Frozen winner replay versus same-module actual v112 CK control;not new full1178 acceptance')
  (out/('%04d_%s.json'%(i,'profile' if args.profile else 'normal'))).write_text(json.dumps(row,indent=2)+'\n');allrows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['checks','selected']}),flush=True)
 finally:ck.probe_destroy(ctx);native.close()
 del a,b,olda,oldb,pb,oldpb,raw,c,desc,cd,vv;gc.collect();torch.cuda.empty_cache()
print('FROZEN_WINNERS_REPLAY_PASS',len(allrows),flush=True)
