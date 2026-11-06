"""CK-owned ELF reader and bounded disassembler, adapted from P6d."""
import collections,hashlib,mmap,re,struct,subprocess,msgpack
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OBJDUMP="/opt/dtk/aillvm/bin/llvm-objdump"

def sections(data):
 assert data[:6]==b'\x7fELF\x02\x01'
 shoff=struct.unpack_from('<Q',data,0x28)[0]
 ent,num,strings=struct.unpack_from('<HHH',data,0x3a)
 secs=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*ent) for i in range(num)]
 names=secs[strings];out=[]
 for i,s in enumerate(secs):
  start=names[4]+s[0];end=data.find(b'\0',start)
  out.append(dict(index=i,name=data[start:end].decode(),typ=s[1],addr=s[3],off=s[4],size=s[5],link=s[6],ent=s[9]))
 return out

def symbols(data,secs):
 result={}
 for s in secs:
  if s['typ'] not in [2,11]:continue
  st=secs[s['link']]
  for j in range(s['size']//(s['ent'] or 24)):
   name,info,other,section,address,size=struct.unpack_from('<IBBHQQ',data,s['off']+j*(s['ent'] or 24))
   if not section or section>=len(secs):continue
   start=st['off']+name;end=data.find(b'\0',start)
   text=data[start:end].decode()
   if info&15 in [2,10]:result[text]=dict(address=address,size=size,section=section)
 return result

def metadata(data,secs):
 result={}
 for s in secs:
  if s['typ']!=7:continue
  pos=s['off'];end=pos+s['size']
  while pos+12<=end:
   n,d,t=struct.unpack_from('<III',data,pos);pos+=12
   name=data[pos:pos+n].rstrip(b'\0');pos+=(n+3)&~3
   desc=data[pos:pos+d];pos+=(d+3)&~3
   if name==b'AMDGPU' and t==32:
    md=msgpack.unpackb(desc,raw=False)
    for kernel in md.get('amdhsa.kernels',[]):result[kernel['.name']]=kernel
 return result

def analyze_text(path):
 instructions=[]
 for line in path.read_text(errors='replace').splitlines():
  m=re.match(r'^\s*([\w.]+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*(.*)',line)
  if m:instructions.append(dict(address=int(m[3],16),opcode=m[1],operands=m[2].strip(),encoding=m[4]))
 counts=collections.Counter(i['opcode'] for i in instructions)
 waits=collections.Counter(i['operands'] for i in instructions if i['opcode']=='s_waitcnt')
 labels=[line.strip() for line in path.read_text().splitlines() if re.match(r'^[0-9a-fA-F]+ <',line)]
 return dict(static_instructions=len(instructions),opcodes=dict(counts),wait_operands=dict(waits),labels=labels,
  direct_lds_loads=sum(i['opcode'].startswith('buffer_load') and ('_lds' in i['opcode'] or re.search(r'\blds\b',i['operands']) is not None) for i in instructions),
  unknown_lines=sum('<unknown>' in line for line in path.read_text().splitlines()))

def dump(path,data,secs,syms,md,name,tag):
 symbol=syms[name];start=symbol['address'];size=symbol['size']
 if not size:
  nexts=[s['address'] for s in syms.values() if s['section']==symbol['section'] and s['address']>start]
  sec=secs[symbol['section']];size=(min(nexts) if nexts else sec['addr']+sec['size'])-start
 sec=secs[symbol['section']];offset=sec['off']+start-sec['addr'];machine=bytes(data[offset:offset+size])
 output=ROOT/'isa'/f'{tag}.s';output.parent.mkdir(exist_ok=True)
 command=[OBJDUMP,'-d','--mcpu=gfx936',f'--start-address={hex(start)}',f'--stop-address={hex(start+size)}',str(path)]
 with output.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.PIPE,text=True)
 assert r.returncode==0,r.stderr
 return dict(tag=tag,symbol=name,code_object=str(path),address=hex(start),bytes=size,sha256=hashlib.sha256(machine).hexdigest(),
  metadata={k:v for k,v in md.get(name,{}).items() if k!='.args'},assembly=str(output),command=command,**analyze_text(output))

def bundles(data):
 magic=b'__CLANG_OFFLOAD_BUNDLE__';pos=0
 while True:
  pos=data.find(magic,pos)
  if pos<0:return
  count=struct.unpack_from('<Q',data,pos+24)[0];p=pos+32
  for i in range(count):
   off,size,n=struct.unpack_from('<QQQ',data,p);p+=24
   triple=data[p:p+n].decode();p+=n
   yield triple,data[pos+off:pos+off+size]
  pos+=32
