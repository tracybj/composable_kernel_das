"""Same-build balanced development screen,not final public-API acceptance."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,resource,gc,hashlib,statistics,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert os.environ['HIP_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
libpath=E/'build/v100/libprobe.so';sha=hashlib.sha256(libpath.read_bytes()).hexdigest();math=json.loads((E/'results/wasp_split_m_v100_math.json').read_text());assert math['status']=='PASS' and len(math['checks'])==212 and math['library_sha256']==sha
lib=C.CDLL(str(libpath));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p;lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
configs=json.loads((Path(__file__).parent/'candidates.json').read_text());current={int(x['ID']):x for x in csv.DictReader((E/'docs/current_cached_v97_1178.csv').open())};ids=[48,96,302,378,448,641,688,705,884,902,1038,1103,1126,1164];out=E/'results/wasp_split_m_v100';out.mkdir(exist_ok=True);assert not (out/'records.jsonl').exists()
manifest=dict(library_sha256=sha,IDs=ids,warmup=2,repeat=5,samples=3,scope='Same-build development screen;no production promotion',ordering='Rotate per case and sample',validation='All output exact FP32 integer reference;256-row bounded comparisons;guards and sampled immutable inputs')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');records=[];started=time.time()
with (out/'records.jsonl').open('w') as stream:
 for ci,idx in enumerate(ids):
  row=current[idx];g,m,n,k=[int(row[x]) for x in ['G','M','N','K']];(out/'active.json').write_text(json.dumps(dict(ID=idx,M=m,N=n,K=k,G=g,stage='allocate')))
  torch.manual_seed(20261030+idx);a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);beg.record();pb=b.reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous();end.record();end.synchronize();prepare_ms=beg.elapsed_time(end)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n);ref=torch.empty_like(c)
  for j in range(g):ref[j]=(a[j].float()@b[j].float().T).to(torch.int32)
  snaps=[]
  for v in (a,b,pb):
   inds=torch.arange(256,device='cuda')*(v.numel()-1)//255;snaps.append((inds,v.reshape(-1)[inds].clone()))
  contexts={};ds=[]
  for packed in [False,True]:
   vv=pb if packed else b;d=(Desc*g)(*[Desc(a[j].data_ptr(),vv[j].data_ptr(),c[j].data_ptr(),1,m,n,k,k,k,n,0,None,None) for j in range(g)]);ds.append(d);contexts[packed]=lib.probe_create(d,g);assert contexts[packed]
  rid=int(row['route']);baseline={1:34001,3:34101,4:34111,5:32001}.get(rid,int(row['legacy_instance']));variants=[dict(id=baseline,packed_weight_B=rid==5,label='current')]
  for cfg in configs:
   mm,nn=(n,m) if cfg['transpose'] else (m,n);bm=128 if cfg['small_M_tile'] else 256;bn=32*cfg.get('consumer_waves',8)
   if mm%bm or nn%bn or k%128:continue
   if cfg['id']==baseline:continue
   variants.append({**cfg,'label':str(cfg['id'])})
  samples={v['id']:[] for v in variants};checked={}
  try:
   for sample in range(3):
    shift=(ci+sample)%len(variants);order=variants[shift:]+variants[:shift]
    for cfg in order:
     cid=cfg['id'];(out/'active.json').write_text(json.dumps(dict(ID=idx,candidate=cid,sample=sample,stage='timing')));c.fill_(123456789);torch.cuda.synchronize();ms=float(lib.probe_run(contexts[cfg['packed_weight_B']],cid,2,5));assert ms>0,(idx,cid,ms);samples[cid].append(ms)
     if sample==2:
      bad=0
      for j in range(g):
       for r0 in range(0,m,256):bad+=int((c[j,r0:r0+256]!=ref[j,r0:r0+256]).sum())
      guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=all(torch.equal(v.reshape(-1)[inds],old) for v,(inds,old) in zip((a,b,pb),snaps));assert bad==0 and guard and unchanged,(idx,cid,bad,guard,unchanged);checked[cid]=True
   for cfg in variants:
    cid=cfg['id'];ms=statistics.median(samples[cid]);tops=2*g*m*n*k/ms/1e9;rec=dict(ID=idx,M=m,N=n,K=k,G=g,candidate=cid,label=cfg['label'],packed=cfg['packed_weight_B'],ms=ms,TOPS=tops,target_TOPS=float(row['BLAS_target_TOPS']),R=tops/float(row['BLAS_target_TOPS']),samples_ms=samples[cid],numeric_status='PASS',packing_event_interval_ms=prepare_ms if cfg['packed_weight_B'] else 0,extra_bytes=pb.numel() if cfg['packed_weight_B'] else 0);records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
  finally:
   for ctx in contexts.values():lib.probe_destroy(ctx)
  part=[x for x in records if x['ID']==idx];control=next(x for x in part if x['label']=='current');best=max(part,key=lambda x:x['TOPS']);print(json.dumps(dict(ID=idx,completed=ci+1,total=len(ids),current_TOPS=control['TOPS'],best_candidate=best['candidate'],best_TOPS=best['TOPS'],speedup=best['TOPS']/control['TOPS'],best_R=best['R'])),flush=True)
  (out/'progress.json').write_text(json.dumps(dict(completed=ci+1,total=len(ids),last_ID=idx,records=len(records))))
  del a,b,pb,raw,c,ref,snaps,ds,v,vv,inds;gc.collect();torch.cuda.empty_cache()
assert hashlib.sha256(libpath.read_bytes()).hexdigest()==sha;(out/'done.json').write_text(json.dumps(dict(status='PASS',shapes=len(ids),records=len(records),elapsed_seconds=time.time()-started,library_sha256=sha))+'\n');print('V100_SCREEN_DONE',len(records),flush=True)
