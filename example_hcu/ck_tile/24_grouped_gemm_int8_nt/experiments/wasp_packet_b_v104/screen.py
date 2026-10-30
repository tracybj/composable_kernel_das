"""Same-build balanced format1/format2 screen; not public-API acceptance."""
from pathlib import Path
import ctypes as C,csv,json,os,sys,resource,gc,hashlib,statistics,time,argparse
resource.setrlimit(resource.RLIMIT_CORE,(0,0));p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');opt=p.parse_args()
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;assert os.environ['HIP_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2];sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
libpath=E/'build/v104/libprobe.so';sha=hashlib.sha256(libpath.read_bytes()).hexdigest();math=json.loads((E/'results/wasp_packet_b_v104_math.json').read_text());assert math['status']=='PASS' and len(math['checks'])==48 and math['library_sha256']==sha
lib=C.CDLL(str(libpath));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p;lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p]
current={int(x['ID']):x for x in csv.DictReader((E/'docs/current_cached_v103_1178.csv').open())}
ids=([i for i,t in current.items() if int(t['M'])>=256] if opt.all else [48,96,302,641,705,902,1038,1126,1164,896,900,1034,1122,298,699,35])
assert all(int(current[i]['M'])%256==0 and int(current[i]['N'])%256==0 and int(current[i]['K'])%128==0 for i in ids)
out=E/'results'/('wasp_packet_b_all_v104' if opt.all else 'wasp_packet_b_v104');out.mkdir(exist_ok=True);assert not (out/'records.jsonl').exists()
(out/'manifest.json').write_text(json.dumps(dict(library_sha256=sha,IDs=ids,warmup=2,repeat=5,samples=3,scope='Development comparison against same-build old M256 packed kernel',ordering='Rotate per shape/sample',reference='[-8,8] integer exact FP32;256-row bounded comparison',format2='G,K128,N32,Nsub2,Ksub2,Kgrp4,Nlane16,Kbyte16'),indent=2)+'\n')
records=[];started=time.time()
with (out/'records.jsonl').open('w') as stream:
 for ci,idx in enumerate(ids):
  row=current[idx];g,m,n,k=[int(row[x]) for x in ['G','M','N','K']];(out/'active.json').write_text(json.dumps(dict(ID=idx,M=m,N=n,K=k,G=g,stage='allocate')))
  free,_=torch.cuda.mem_get_info();estimate=g*(m*k+3*n*k+4*(m*n+32))+4*(m*k+n*k)+12*m*n+2*1024**3;assert estimate<free,('Keep original G',idx,estimate,free)
  torch.manual_seed(20261030+idx);a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
  packed={};prep={}
  for fmt in [1,2]:
   beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);beg.record()
   packed[fmt]=(b.reshape(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous() if fmt==1 else b.reshape(g,n//32,2,16,k//128,2,4,16).permute(0,4,1,2,5,6,3,7).contiguous())
   end.record();end.synchronize();prep[fmt]=beg.elapsed_time(end)
  raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n);snaps=[]
  for v in [a,b,packed[1],packed[2]]:
   inds=torch.arange(256,device='cuda')*(v.numel()-1)//255;snaps.append((inds,v.reshape(-1)[inds].clone()))
  contexts={};ds=[]
  for fmt in [1,2]:
   vv=packed[fmt];d=(Desc*g)(*[Desc(a[j].data_ptr(),vv[j].data_ptr(),c[j].data_ptr(),1,m,n,k,k,k,n,0,None,None) for j in range(g)]);ds.append(d);contexts[fmt]=lib.probe_create(d,g);assert contexts[fmt]
  variants=[32001,37001,37002,37008];samples={v:[] for v in variants}
  try:
   for sample in range(3):
    shift=(ci+sample)%4;order=variants[shift:]+variants[:shift]
    for cid in order:
     fmt=1 if cid==32001 else 2;(out/'active.json').write_text(json.dumps(dict(ID=idx,candidate=cid,sample=sample,stage='timing')));c.fill_(123456789);torch.cuda.synchronize();ms=float(lib.probe_run(contexts[fmt],cid,2,5));assert ms>0,(idx,cid,ms);samples[cid].append(ms)
     if sample==2:
      bad=0
      for j in range(g):
       af,bf=a[j].float(),b[j].float();reference=(af@bf.T).to(torch.int32)
       for r0 in range(0,m,256):bad+=int((c[j,r0:r0+256]!=reference[r0:r0+256]).sum())
       del af,bf,reference
      guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all());unchanged=all(torch.equal(v.reshape(-1)[inds],old) for v,(inds,old) in zip([a,b,packed[1],packed[2]],snaps));assert bad==0 and guard and unchanged,(idx,cid,bad,guard,unchanged)
   for cid in variants:
    fmt=1 if cid==32001 else 2;ms=statistics.median(samples[cid]);tops=2*g*m*n*k/ms/1e9;rec=dict(ID=idx,M=m,N=n,K=k,G=g,candidate=cid,format=fmt,ms=ms,TOPS=tops,target_TOPS=float(row['BLAS_target_TOPS']),R=tops/float(row['BLAS_target_TOPS']),samples_ms=samples[cid],numeric_status='PASS',packing_event_interval_ms=prep[fmt],extra_bytes=packed[fmt].numel());records.append(rec);stream.write(json.dumps(rec)+'\n');stream.flush();os.fsync(stream.fileno())
  finally:
   for ctx in contexts.values():lib.probe_destroy(ctx)
  part=[x for x in records if x['ID']==idx];control=next(x for x in part if x['candidate']==32001);best=max(part,key=lambda x:x['TOPS']);print(json.dumps(dict(ID=idx,completed=ci+1,total=len(ids),control_TOPS=control['TOPS'],best_candidate=best['candidate'],best_TOPS=best['TOPS'],speedup=best['TOPS']/control['TOPS'],best_R=best['R'])),flush=True)
  (out/'progress.json').write_text(json.dumps(dict(completed=ci+1,total=len(ids),last_ID=idx,records=len(records))))
  del a,b,packed,raw,c,snaps,ds,v,vv,inds;gc.collect();torch.cuda.empty_cache()
assert hashlib.sha256(libpath.read_bytes()).hexdigest()==sha;(out/'done.json').write_text(json.dumps(dict(status='PASS',shapes=len(ids),records=len(records),elapsed_seconds=time.time()-started,library_sha256=sha))+'\n');print('V104_SCREEN_DONE',len(records),flush=True)
