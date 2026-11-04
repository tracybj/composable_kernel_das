#!/usr/bin/env python3
"""Verify original shaders unchanged and isolate experimental SLC machine bits."""
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('isa_helper','/workspace/v7_coverage/p6d/elf_isa.py')
elf=importlib.util.module_from_spec(spec);spec.loader.exec_module(elf)

def kernels(path):
 result={}
 with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as host:
  fat=next(s for s in elf.sections(host) if s['name']=='.hip_fatbin')
  for triple,data in elf.bundles(host[fat['off']:fat['off']+fat['size']]):
   if 'gfx936' not in triple or data[:4]!=b'\x7fELF':continue
   secs=elf.sections(data);symbols=elf.symbols(data,secs);md=elf.metadata(data,secs)
   for name,s in symbols.items():
    if 'kentry' not in name or 'GroupedGemmV7UniformKernel' not in name:continue
    sec=secs[s['section']];offset=sec['off']+s['address']-sec['addr'];size=s['size']
    if not size:
     following=[v['address'] for v in symbols.values() if v['section']==s['section'] and v['address']>s['address']]
     size=(min(following) if following else sec['addr']+sec['size'])-s['address']
    code=bytes(data[offset:offset+size]);normalized=bytearray(code);count=0
    for off in range(0,len(code)-3,4):
     if struct.unpack_from('<I',code,off)[0]==0xe05f3000:
      struct.pack_into('<I',normalized,off,0xe05d3000);count+=1
    demangled=subprocess.check_output(['/opt/dtk/aillvm/bin/llvm-cxxfilt',name],text=True).strip()
    result[name]=dict(symbol=name,demangled=demangled,bytes=len(code),sha256=hashlib.sha256(code).hexdigest(),without_slc_sha256=hashlib.sha256(normalized).hexdigest(),slc_idxen_loads=count,metadata={k:v for k,v in md[name].items() if k not in ['.args','.unfolded_args']})
 return result

def main():
 old=kernels(ROOT/'bench_before')
 new=kernels(Path('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark'))
 preserved=[]
 for name,r in old.items():
  assert name in new,name
  assert r['sha256']==new[name]['sha256'],('Original shader changed',name)
  preserved.append(dict(symbol=name,sha256=r['sha256']))
 pairs=[]
 for name,r in new.items():
  if 'GemmPipelineV7CacheNoSlc' not in r['demangled']:continue
  prefix=r['demangled'].split('GemmPipelineV7CacheNoSlc')[0]
  m01=int(re.search(r'false, false>, 1, (\d+)>',prefix)[1])
  matches=[b for b in old.values() if 'GemmPipelineV7Uniform<' in b['demangled'] and '(ck_tile::V7UniformKind)0' in b['demangled'] and re.search(r'false, false>, 1, '+str(m01)+'>',b['demangled'].split('GemmPipelineV7Uniform')[0])]
  assert len(matches)==1,(m01,len(matches))
  b=matches[0];isolated=r['without_slc_sha256']==b['without_slc_sha256']
  pairs.append(dict(m01=m01,original=b,experiment=r,only_slc_machine_bits_differ=isolated))
  print('CACHE_ISA',m01,'only_slc_bits',isolated,'bytes',b['bytes'],r['bytes'],'slc_loads',b['slc_idxen_loads'],r['slc_idxen_loads'])
 assert len(pairs)==2,len(pairs)
 resources=[]
 for row in new.values():
  if not any(x in row['demangled'] for x in ['GemmPipelineV7SmallMDrain<','GemmPipelineV7SmallMK32<','GemmPipelineV7Square128<']):continue
  v='drain' if 'SmallMDrain<' in row['demangled'] else 'square128' if 'Square128<' in row['demangled'] else 'k32n64' if 'V7SmallMK32Geometry<64>' in row['demangled'] else 'k32n128'
  resources.append(dict(variant=v,declared_dynamic_lds_bytes=49152 if v=='k32n128' else 32768,**row))
 (ROOT/'small_resources.json').write_text(json.dumps(resources,indent=2))
 print('SMALL_SHADER_RESOURCES',len(resources))
 (ROOT/'cache_isa.json').write_text(json.dumps(dict(original_shaders_unchanged=preserved,cache_pairs=pairs,small_resources=resources),indent=2))
 print('ORIGINAL_SHADER_PRESERVATION_PASS',len(preserved))

if __name__=='__main__':main()
