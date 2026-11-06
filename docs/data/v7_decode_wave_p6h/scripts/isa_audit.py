"""Verify actual new instructions and resource metadata; CPU-only."""
import sys,mmap,json,hashlib,re
from pathlib import Path
import elf_reader as e
ROOT=Path(__file__).resolve().parent
p=Path(sys.argv[1]);tag=sys.argv[2];rows=[]
with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as host:
 fat=next(s for s in e.sections(host) if s['name']=='.hip_fatbin')
 for triple,data in e.bundles(host[fat['off']:fat['off']+fat['size']]):
  if 'gfx936' not in triple or data[:4]!=b'\x7fELF':continue
  secs=e.sections(data);syms=e.symbols(data,secs);md=e.metadata(data,secs)
  names=[n for n in md if 'DecodeNnWaveKernel' in n]
  if not names:continue
  co=ROOT/(tag+'_gfx936.co');co.write_bytes(data)
  for name in names:
   match=re.search(r'DecodeNnWaveKernelI((?:Li\d+E)+)E',name)
   parameters=[int(v) for v in re.findall(r'Li(\d+)E',match[1])] if match else [2]
   depth=parameters[-1];rows_count=parameters[0] if len(parameters)>1 else 1
   row=e.dump(co,data,secs,syms,md,name,tag+'_rows'+str(rows_count)+'_depth'+str(depth))
   assert row['unknown_lines']==0
   assert row['opcodes'].get('s_barrier',0)==0
   assert row['metadata'].get('.private_segment_fixed_size',0)==0
   assert row['metadata'].get('.sgpr_spill_count',0)==0
   assert row['metadata'].get('.vgpr_spill_count',0)==0
   assert row['direct_lds_loads']>0
   a_lds=row['opcodes'].get('buffer_load_ushort',0)>0
   row['pipeline']='A_LDS_B_VGPR' if a_lds else 'A_VGPR_B_LDS'
   row['declared_dynamic_lds_bytes']=depth*4096*(rows_count if a_lds else 1);row['depth']=depth;row['row_fragments']=rows_count;rows.append(row)
assert len(rows) in [1,2,3,6],len(rows)
(ROOT/(tag+'_isa.json')).write_text(json.dumps(rows,indent=2));print('DECODE_ISA_PASS',json.dumps([dict(row_fragments=x['row_fragments'],depth=x['depth'],bytes=x['bytes'],vgpr=x['metadata']['.vgpr_count'],sgpr=x['metadata']['.sgpr_count'],scratch=x['metadata']['.private_segment_fixed_size'],lds=x['declared_dynamic_lds_bytes']) for x in rows]))
