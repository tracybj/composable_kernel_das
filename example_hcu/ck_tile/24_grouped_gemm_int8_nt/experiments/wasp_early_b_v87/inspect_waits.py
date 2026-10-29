from pathlib import Path
import subprocess,json,struct,hashlib,re,collections
e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt');out=e/'build/v87/isa_compare';out.mkdir(exist_ok=True)
lib=e/'build/v87/libprobe.so';data=lib.read_bytes();sha=hashlib.sha256(data).hexdigest();magic=b'__CLANG_OFFLOAD_BUNDLE__';assert data.count(magic)==1
beg=data.index(magic);pos=beg+len(magic);count=struct.unpack_from('<Q',data,pos)[0];pos+=8;objects=[]
for i in range(count):
 off,size,n=struct.unpack_from('<QQQ',data,pos);pos+=24;name=data[pos:pos+n].decode();pos+=n
 if 'gfx936' in name:objects.append(data[beg+off:beg+off+size])
assert len(objects)==1 and objects[0][:4]==b'\x7fELF';co=out/'device.co';co.write_bytes(objects[0])
res=json.loads((e/'build/v87/resources.json').read_text());results=[]
for early in [0,1]:
 x=next(x for x in res['kernels'] if x['params']==[0,8,4,4,1,1,1024,early]);sym=x['kernel_symbol'];path=out/('early.s' if early else 'late.s')
 q=subprocess.run(['/opt/dtk/aillvm/bin/llvm-objdump','--disassemble','--disassemble-symbols='+sym,str(co)],capture_output=True,text=True,check=True);path.write_text(q.stdout)
 instructions=[]
 for line in q.stdout.splitlines():
  s=line.split('//')[0].strip()
  if re.match(r'^(?:s_|v_|buffer_|global_|flat_|ds_)',s):instructions.append(s)
 kinds=[]
 for i,s in enumerate(instructions):
  kind=''
  if 'v_mmac_' in s:kind='MMAC'
  elif 'buffer_load_' in s:kind='VMEM_A' if 'lds' in s else 'VMEM_B' if 'idxen' in s else 'VMEM_other'
  elif 'ds_read_' in s:kind='LDS_READ'
  elif 's_waitcnt' in s:kind='WAIT:'+s.split('s_waitcnt',1)[1].strip()
  elif 's_barrier' in s:kind='BARRIER'
  if kind:kinds.append((i,kind))
 first=next(i for i,(_,k) in enumerate(kinds) if k=='MMAC');sample=kinds[max(0,first-20):first+320];rle=[]
 for inst,k in sample:
  if rle and rle[-1]['kind']==k:rle[-1]['count']+=1
  else:rle.append(dict(instruction=inst,kind=k,count=1))
 record=dict(early_B=bool(early),kernel_symbol=sym,isa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),static_counts=dict(collections.Counter(k for _,k in kinds)),first_compute_sequence_rle=rle,isa_path=str(path));results.append(record)
 print('EARLY',early,'STATIC_COUNTS',json.dumps(record['static_counts']));print('FIRST_COMPUTE_RLE',json.dumps(rle[:70]))
assert hashlib.sha256(lib.read_bytes()).hexdigest()==sha
(out/'wait_analysis.json').write_text(json.dumps(dict(library_sha256=sha,records=results,scope='Static generated instruction order/counts across the kernel including fallback/tail; not dynamic counters or cycle attribution'),indent=2))
