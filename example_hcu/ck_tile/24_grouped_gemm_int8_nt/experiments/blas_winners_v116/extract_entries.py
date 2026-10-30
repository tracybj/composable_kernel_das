from pathlib import Path
import subprocess,json,hashlib,struct,collections
import msgpack
e=Path('/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt');b=e/'build/blas_winners_v116';b.mkdir(exist_ok=True);manifest=json.loads((e/'experiments/blas_winners_v116/winners.json').read_text());root=Path('/workspace/GroupGemm/baseline/hipblaslt-install');names=sorted({x['kernel_name'] for x in manifest['results']});found=[]
def metadata(data):
 assert data[:6]==b'\x7fELF\x02\x01';off=struct.unpack_from('<Q',data,40)[0];ents,num=struct.unpack_from('<HH',data,58);out=[]
 for i in range(num):
  sh=struct.unpack_from('<IIQQQQIIQQ',data,off+i*ents)
  if sh[1]!=7:continue
  raw=data[sh[4]:sh[4]+sh[5]];pos=0
  while pos+12<=len(raw):
   ns,ds,k=struct.unpack_from('<III',raw,pos);pos+=12;name=raw[pos:pos+ns].rstrip(b'\0');pos=(pos+ns+3)&~3;desc=raw[pos:pos+ds];pos=(pos+ds+3)&~3
   if name==b'AMDGPU':out+=msgpack.unpackb(desc,raw=False,strict_map_key=False).get('amdhsa.kernels',[])
 return out
for p in sorted((root/'lib/hipblaslt').rglob('*')):
 if not p.is_file() or p.suffix not in ['.co','.hsaco'] or 'gfx936' not in str(p):continue
 data=p.read_bytes()
 hits=[n for n in names if n.encode() in data]
 if not hits:continue
 syms=subprocess.check_output(['readelf','-Ws',str(p)],text=True)
 for n in hits:
  symbols={l.split()[-1] for l in syms.splitlines() if l.split() and (l.split()[-1]==n or l.split()[-1]==n+'.kd')}
  if n not in symbols:continue
  i=next(x['algorithm_id'] for x in manifest['results'] if x['kernel_name']==n);out=b/str(i);out.mkdir(exist_ok=True)
  q=subprocess.run(['/opt/dtk/aillvm/bin/llvm-objdump','-d','--mcpu=gfx936','--disassemble-symbols='+n,str(p)],capture_output=True,text=True);assert q.returncode==0,q.stderr
  assert len(q.stdout)>1000,(i,q.stdout[:500]);(out/'kernel.isa').write_text(q.stdout)
  md=[x for x in metadata(data) if x.get('.name')==n];assert len(md)==1,(i,len(md));(out/'metadata.json').write_text(json.dumps(md[0],indent=2)+'\n')
  counts=collections.Counter()
  for line in q.stdout.splitlines():
   if '//' not in line:continue
   v=line.split('//',1)[0].strip().split()
   if v:counts[v[0]]+=1
  row=dict(algorithm_id=i,kernel=n,object=str(p),object_sha256=hashlib.sha256(data).hexdigest(),static_instruction_counts=dict(counts),metadata=md[0],isa_sha256=hashlib.sha256(q.stdout.encode()).hexdigest())
  found.append(row);print(json.dumps(dict(algorithm_id=i,object=str(p),resources={k:v for k,v in md[0].items() if k in ['.vgpr_count','.sgpr_count','.private_segment_fixed_size','.vgpr_spill_count','.group_segment_fixed_size','.max_flat_workgroup_size','.wavefront_size','.kernarg_segment_size']},key_instructions={k:v for k,v in counts.items() if k.startswith(('v_mmac','v_dot','v_mfma','buffer_','ds_','s_barrier','s_waitcnt','.long'))})),flush=True)
assert {x['kernel'] for x in found}==set(names),(set(names)-{x['kernel'] for x in found})
(b/'extracted.json').write_text(json.dumps(dict(blas_library=str((root/'lib/libhipblaslt.so').resolve()),blas_sha256=hashlib.sha256((root/'lib/libhipblaslt.so').read_bytes()).hexdigest(),kernels=found),indent=2)+'\n');print('EXTRACTED',len(found))
