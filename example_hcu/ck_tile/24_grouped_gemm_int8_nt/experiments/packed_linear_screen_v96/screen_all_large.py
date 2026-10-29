"""All344 original large-M shapes:raw controls/linear B and cached INT8 weight packing.

Development screening only. Existing FP16/BF16 interfaces and source remain unchanged.
"""
from pathlib import Path
import ctypes as C,csv,json,os,sys,statistics,gc,time,resource,hashlib,subprocess,traceback
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
assert os.environ['HIP_VISIBLE_DEVICES']=='7' and os.environ['CUDA_VISIBLE_DEVICES']=='7'
E=Path(__file__).resolve().parents[2];out=E/'results/packed_linear_screen_v96';out.mkdir(exist_ok=True)
resume='--resume' in sys.argv
assert (out/'records.jsonl').exists()==resume,'Use explicit resume only for verified prior records'
sys.path.insert(0,'/workspace/GroupGemm/baseline/int8_nt');from worker import Desc
cases=[x for x in csv.DictReader((E/'docs/targets_int8_nt.csv').open()) if int(x['M'])>=256]
formal={int(x['ID']):x for x in csv.DictReader((E/'docs/current_public_v80_1178.csv').open(encoding='utf-8-sig'))}
assert len(cases)==344 and all(int(x['M'])%256==int(x['N'])%256==int(x['K'])%128==0 and int(x['K'])<16384 for x in cases)
paths={'raw':E/'build/v92/libprobe.so','packed':E/'build/v90/libprobe.so'}
expected={'raw':'41bc0ef7dba343131d43f5ce8c6990d03fc22d2d9eb7bf3ca328ac45d7d1daf4','packed':'ad4847aa77a0707b5753783b42de6043d143e4bbe8aa4297ce499f9120df4de1'}
libs={}
for role,p in paths.items():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[role]
 lib=C.CDLL(str(p));lib.probe_create.argtypes=[C.POINTER(Desc),C.c_int];lib.probe_create.restype=C.c_void_p
 lib.probe_run.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.probe_run.restype=C.c_float;lib.probe_destroy.argtypes=[C.c_void_p];libs[role]=lib
context=json.loads((E/'build/public_v80/sources.json').read_text())
for p,h in context.items():assert hashlib.sha256((E.parents[2]/p).read_bytes()).hexdigest()==h
for v,count in [(90,96),(92,300)]:
 a=json.loads((E/f'results/d2l_v{v}_math.json').read_text());assert len(a)==count and all(x['status']=='PASS' for x in a)
manifest=dict(gpu=7,total_shapes=344,candidates_per_shape=6,total_records=2064,library_sha256=expected,existing_public_sources_unchanged=context,source_revision=subprocess.check_output(['git','-C',str(E.parents[2]),'rev-parse','HEAD'],text=True).strip(),warmup=2,repeat=5,samples=3,ordering='rotate candidate order per shape and per sample',dtype='INT8xINT8->INT32_no_scale',layout='NT',scope='All344 originalM>=256 dimensions/groups;current public default vs4 raw linear policies vs prepacked original weight B;development screen,not1178 public acceptance',weight_preparation='User-authorized one-time INT8 weight loading/reuse;cost andextra bytes recorded separately;original activation A is never prepacked',precision_isolation='No existing FP16/BF16 ABI,struct,layout ordefault selector modifications',validation='Every output exact FP32 reference withTF32off,inputs[-8,8];full-range checks separate96+300;output guards and256sample input immutability per operand')
manifest['output_comparison']='Full output,256-row chunks per group;bounded reduction temporaries'
if not resume:(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
else:
 original=json.loads((out/'manifest.json').read_text());assert original['library_sha256']==expected and original['total_shapes']==344
 manifest['original_manifest_sha256']=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
 (out/'resume_manifest_01.json').write_text(json.dumps(manifest,indent=2)+'\n')
def atomic(name,value):
 p=out/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)
def prepare_weight(b):
 g,n,k=b.shape;view=b.view(g,n//16,16,k//64,4,16).permute(0,3,1,4,2,5)
 dst=torch.empty(view.shape,device='cuda',dtype=torch.int8)
 for _ in range(2):dst.copy_(view)
 samples=[]
 for _ in range(3):
  beg,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);beg.record()
  for _ in range(5):dst.copy_(view)
  end.record();end.synchronize();samples.append(beg.elapsed_time(end)/5)
 return dst,dict(ms=statistics.median(samples),samples_ms=samples,extra_bytes=dst.numel(),method='Preallocated GPU permuted-view copy;warmup2/repeat5/sample3;GPU time,allocation excluded')
records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()] if resume else []
assert len(records)%6==0
resume_count=len(records)//6
for i in range(resume_count):
 part=records[i*6:i*6+6];case=cases[i]
 assert all(x['ID']==int(case['ID']) and x['bad_elements']==0 and x['guard'] and x['input_samples_unchanged'] for x in part)
 assert {x['candidate_name'] for x in part}=={'public_v80','linear_original','linear_original_record1024','linear_transposed','linear_transposed_record1024','cached_weight_original'}
 assert all(all(x[t]==int(case[t]) for t in ['M','N','K','G']) for x in part)
previous_elapsed=json.loads((out/'error_initial_oom.json').read_text())['elapsed_seconds'] if resume else 0
started=time.time()
try:
 with (out/'records.jsonl').open('a' if resume else 'w') as stream:
  for case_number,case in enumerate(cases[resume_count:],resume_count):
   g,m,n,k=[int(case[x]) for x in ('G','M','N','K')];idx=int(case['ID']);phase=formal[idx]['phase']
   atomic('active.json',dict(ID=idx,M=m,N=n,K=k,G=g,phase=phase,stage='allocate'))
   free,_=torch.cuda.mem_get_info();estimate=g*(m*k+2*n*k+8*m*n+128)+8*(m*k+n*k+m*n)+9*256*n+1024**3
   assert estimate<free,('Memory budget;do not change G',idx,estimate,free)
   torch.manual_seed(20261029+idx)
   a=torch.randint(-8,9,(g,m,k),device='cuda',dtype=torch.int8);b=torch.randint(-8,9,(g,n,k),device='cuda',dtype=torch.int8)
   pb,pack=prepare_weight(b)
   raw=torch.full((g,m*n+32),117,device='cuda',dtype=torch.int32);c=raw[:,16:-16].view(g,m,n);reference=torch.empty_like(c)
   atomic('active.json',dict(ID=idx,M=m,N=n,K=k,G=g,phase=phase,stage='reference'))
   for group in range(g):reference[group]=(a[group].float()@b[group].float().T).to(torch.int32)
   snapshots=[]
   for operand in (a,b,pb):
    indices=torch.arange(256,device='cuda',dtype=torch.int64)*(operand.numel()-1)//255;snapshots.append((indices,operand.reshape(-1)[indices].clone()))
   raw_ds=(Desc*g)(*[Desc(a[i].data_ptr(),b[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
   packed_ds=(Desc*g)(*[Desc(a[i].data_ptr(),pb[i].data_ptr(),c[i].data_ptr(),1,m,n,k,k,k,n,0,None,None) for i in range(g)])
   contexts={'raw':libs['raw'].probe_create(raw_ds,g),'packed':libs['packed'].probe_create(packed_ds,g)};assert all(contexts.values())
   variants=[('public_v80','raw',int(formal[idx]['instance_id'])),('linear_original','raw',34001),('linear_original_record1024','raw',34011),('linear_transposed','raw',34101),('linear_transposed_record1024','raw',34111),('cached_weight_original','packed',32001)]
   ts={name:[] for name,_,_ in variants};checks={}
   try:
    for sample in range(3):
     shift=(case_number+sample)%len(variants);order=variants[shift:]+variants[:shift]
     for name,role,candidate in order:
      atomic('active.json',dict(ID=idx,M=m,N=n,K=k,G=g,phase=phase,stage='timing',sample=sample,candidate=name))
      c.fill_(123456789);torch.cuda.synchronize();ms=float(libs[role].probe_run(contexts[role],candidate,2,5));assert ms>0,(idx,name,ms);ts[name].append(ms)
      if sample==2:
       bad=0
       for check_group in range(g):
        for row0 in range(0,m,256):bad+=int((c[check_group,row0:row0+256]!=reference[check_group,row0:row0+256]).sum())
       guard=bool((raw[:,:16]==117).all()&(raw[:,-16:]==117).all())
       unchanged=all(torch.equal(v.reshape(-1)[indices],old) for v,(indices,old) in zip((a,b,pb),snapshots));assert bad==0 and guard and unchanged,(idx,name,bad,guard,unchanged)
       checks[name]=dict(bad_elements=bad,guard=guard,input_samples_unchanged=unchanged)
    for name,role,candidate in variants:
     ms=statistics.median(ts[name]);rec=dict(ID=idx,phase=phase,M=m,N=n,K=k,G=g,candidate_name=name,candidate=candidate,library_role=role,ms=ms,samples_ms=ts[name],TOPS=2*g*m*n*k/(ms*1e9),target_TOPS=float(case['target_TOPS']),R=2*g*m*n*k/(ms*1e9)/float(case['target_TOPS']),packing_GPU_ms=pack['ms'] if role=='packed' else 0,extra_weight_bytes=pack['extra_bytes'] if role=='packed' else 0,**checks[name])
     stream.write(json.dumps(rec)+'\n');stream.flush();records.append(rec)
   finally:
    for role,ctx in contexts.items():libs[role].probe_destroy(ctx)
   best=max(records[-6:],key=lambda x:x['TOPS']);progress=dict(completed_shapes=case_number+1,total_shapes=344,records=len(records),last_ID=idx,best=best['candidate_name'],best_R=best['R'],elapsed_seconds=previous_elapsed+time.time()-started)
   atomic('progress.json',progress);print(json.dumps(progress),flush=True)
   atomic('last_packing.json',dict(ID=idx,**pack))
   del a,b,pb,c,raw,reference,raw_ds,packed_ds,snapshots,operand,indices;gc.collect();torch.cuda.empty_cache()
 for role,p in paths.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[role]
 for p,h in context.items():assert hashlib.sha256((E.parents[2]/p).read_bytes()).hexdigest()==h
 assert len(records)==2064
 atomic('done.json',dict(status='PASS',shapes=344,records=2064,scope='Development screening;not final1178 cached/default ABI acceptance',elapsed_seconds=previous_elapsed+time.time()-started));print('V96_ALL_LARGE_DONE',flush=True)
except Exception as exc:
 atomic('error.json',dict(error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=previous_elapsed+time.time()-started));raise
