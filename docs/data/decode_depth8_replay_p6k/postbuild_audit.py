"""CPU-only identity, machine-code, and protected-worktree audit."""
import hashlib,json,mmap,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=Path('/workspace/composable_kernel')
sys.path.insert(0,'/workspace/v7_coverage/p6i_b_vector')
import audit_isa as a
import elf_reader as e
e.ROOT=ROOT
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path('/workspace/v7_coverage/p6j_areg_scope/official_benchmark_before_scope')
new=REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
assert sha(old)=='4ac20b152ba54ac87e6d0214881ee2e8875734ad86663b971d400216e0c34052'
before,_=a.kernels(old);after,_=a.kernels(new)
changed=[dict(target=key[0],symbol=key[1],before=value,after=after.get(key)) for key,value in before.items() if after.get(key)!=value]
assert not changed,changed
selected=[]
with new.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as host:
 fat=next(s for s in e.sections(host) if s['name']=='.hip_fatbin')
 for triple,data in e.bundles(host[fat['off']:fat['off']+fat['size']]):
  if data[:4]!=b'\x7fELF' or 'gfx936' not in triple:continue
  secs=e.sections(data);md=e.metadata(data,secs)
  names=[n for n in md if 'DecodeNnDepth8Kernel' in n]
  if not names:continue
  co=ROOT/'depth8_gfx936.co';co.write_bytes(data);syms=e.symbols(data,secs)
  for name in names:
   row=e.dump(co,data,secs,syms,md,name,'depth8_6003')
   for key in ['.private_segment_fixed_size','.sgpr_spill_count','.vgpr_spill_count']:assert row['metadata'].get(key,0)==0
   assert row['opcodes'].get('s_barrier',0)==0 and row['direct_lds_loads']>0 and row['unknown_lines']==0
   assert row['sha256']=='e252c83b232adef02ccbe90f2e009e601803d14bdef5928a15113ecd2e5ce925',row['sha256']
   row['dynamic_lds_bytes']=32768;selected.append(row)
assert len(selected)==1
initial=json.loads((ROOT/'initial_files_sha256.json').read_text())
shared=set(json.loads((ROOT/'integration_files.json').read_text()))
untouched=[]
for rel,digest in initial.items():
 if rel in shared:continue
 assert sha(REPO/rel)==digest,('PREEXISTING_FILE_CHANGED',rel)
 untouched.append(rel)
source_files=sorted(shared|set(initial)|{'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_decode_depth8_selector_test.cpp'})
record=dict(ok=True,benchmark_sha256=sha(new),initial_binary_sha256=sha(old),
 preserved_control_shaders=[dict(target=k[0],symbol=k[1],sha256=v) for k,v in before.items()],
 new_shader=selected[0],protected_preexisting_files_unchanged=untouched,
 worktree_source_sha256={p:sha(REPO/p) for p in source_files})
(ROOT/'postbuild_audit.json').write_text(json.dumps(record,indent=2))
print('P6K_MACHINE_CODE_PRESERVATION_PASS old_shaders='+str(len(before))+' depth8_sha='+selected[0]['sha256']+' old_protected_files='+str(len(untouched)))
