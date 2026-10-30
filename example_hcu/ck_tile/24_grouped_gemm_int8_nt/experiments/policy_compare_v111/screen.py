"""Same-build balanced development screen,not final public-API acceptance."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,resource,gc,hashlib,statistics,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert os.environ['HIP_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
libpath=E/'build/v111/libprobe.so';sha=hashlib.sha256(libpath.read_bytes()).hexdigest();math=json.loads((E/'results/policy_compare_v111_math.json').read_text());assert math['status']=='PASS' and len(math['checks'])==150 and math['library_sha256']==sha
lib=C.CDLL(str(libpath));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p;lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
configs=[dict(id=39002,packed_weight_B=True)];current={int(x['ID']):x for x in csv.DictReader((E/'docs/current_cached_v106_1178.csv').open())};ids=[i for i,t in current.items() if int(t['M'])>=256];assert len(ids)==344;out=E/'results/policy_compare_v111';out.mkdir(exist_ok=True);assert not (out/'records.jsonl').exists()
manifest=dict(library_sha256=sha,IDs=ids,warmup=2,repeat=5,samples=3,scope='Same-module actual v106 selected route versus newN384;no production promotion',ordering='Rotate per case and sample',validation='All output exact FP32 integer reference;256-row bounded comparisons;guards and sampled immutable inputs')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');records=[];started=time.time()
with (out/'records.jsonl').open('w') as stream:
 for ci,idx in enumerate(ids):
  row=current[idx];g,m,n,k=[int(row[x]) for x in ['G','M','N','K']];(out/'active.json').write_text(json.dumps(dict(ID=idx,M=m,N=n,K=k,G=g,stage='allocate')))
  packable=n%16==0 and k%64==0;free,_=torch.cuda.mem_get_info();estimate=g*(m*k+n*k*(1+int(packable))+4*(m*n+32))+4*(m*k+n*k)+12*m*n+2*1024**3;assert estimate<free,('Keep original G',idx,estimate,free)
  torch.manual_seed(20261030+idx);a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  pb=None;prepare_ms=0
  if packable:
   beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);beg.record();pb=b.reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous();end.record();end.synchronize();prepare_ms=beg.elapsed_time(end)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n)
  snaps=[]
  for v in [x for x in (a,b,pb) if x is not None]:
   inds=torch.arange(256,device='cuda')*(v.numel()-1)//255;snaps.append((inds,v.reshape(-1)[inds].clone()))
  contexts={};ds=[]
  for packed in [False,True]:
   if packed and pb is None:continue
   vv=pb if packed else b;d=(Desc*g)(*[Desc(a[j].data_ptr(),vv[j].data_ptr(),c[j].data_ptr(),1,m,n,k,k,k,n,0,None,None) for j in range(g)]);ds.append(d);contexts[packed]=lib.probe_create(d,g);assert contexts[packed]
  rid=int(row['route']);baseline=50000+rid if rid else int(row['legacy_instance']);variants=[dict(id=baseline,packed_weight_B=rid in [5,6,7,8,9],label='current')]
  for cfg in configs:
   assert packable and m%256==0 and n%256==0 and k%128==0
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
       af,bf=a[j].float(),b[j].float();reference=(af@bf.T).to(torch.int32)
       for r0 in range(0,m,256):bad+=int((c[j,r0:r0+256]!=reference[r0:r0+256]).sum())
       del af,bf,reference
      guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=all(torch.equal(v.reshape(-1)[inds],old) for v,(inds,old) in zip([x for x in (a,b,pb) if x is not None],snaps));assert bad==0 and guard and unchanged,(idx,cid,bad,guard,unchanged);checked[cid]=True
   for cfg in variants:
    cid=cfg['id'];ms=statistics.median(samples[cid]);tops=2*g*m*n*k/ms/1e9;rec=dict(ID=idx,M=m,N=n,K=k,G=g,candidate=cid,label=cfg['label'],packed=cfg['packed_weight_B'],ms=ms,TOPS=tops,target_TOPS=float(row['BLAS_target_TOPS']),R=tops/float(row['BLAS_target_TOPS']),samples_ms=samples[cid],numeric_status='PASS',packing_event_interval_ms=prepare_ms if cfg['packed_weight_B'] else 0,extra_bytes=pb.numel() if cfg['packed_weight_B'] else 0);records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush()
  finally:
   for ctx in contexts.values():lib.probe_destroy(ctx)
  part=[x for x in records if x['ID']==idx];control=next(x for x in part if x['label']=='current');best=max(part,key=lambda x:x['TOPS']);print(json.dumps(dict(ID=idx,completed=ci+1,total=len(ids),current_TOPS=control['TOPS'],best_candidate=best['candidate'],best_TOPS=best['TOPS'],speedup=best['TOPS']/control['TOPS'],best_R=best['R'])),flush=True)
  (out/'progress.json').write_text(json.dumps(dict(completed=ci+1,total=len(ids),last_ID=idx,records=len(records))))
  del a,b,pb,raw,c,snaps,ds,v,vv,inds;gc.collect();torch.cuda.empty_cache()
assert hashlib.sha256(libpath.read_bytes()).hexdigest()==sha;(out/'done.json').write_text(json.dumps(dict(status='PASS',shapes=len(ids),records=len(records),elapsed_seconds=time.time()-started,library_sha256=sha))+'\n');print('V111_ALL344_DONE',len(records),flush=True)
