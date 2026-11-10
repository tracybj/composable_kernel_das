"""Audit an archived official benchmark without rebuilding or changing kernels."""
import hashlib,json,mmap,shutil,sys
from pathlib import Path
sys.path.insert(0,'/workspace/v7_coverage/p6i_b_vector')
import audit_isa as audit
import runtime as r

ROOT=Path(__file__).resolve().parent
old=Path('/workspace/v7_coverage/p6h_decode_arch/trial03/benchmark')
current=Path('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
expected='fe5b5c7402683c0a6459beec4126ec8609ff0bb8bed8b1d114f0f5fdf44997a8'
assert r.sha(old)==expected
old_hashes,_=audit.kernels(old)
current_hashes,_=audit.kernels(current)
preserved=[]
for key,digest in old_hashes.items():
    if 'DecodeNn' in key[1]:continue
    assert current_hashes.get(key)==digest,('OLD_V7_CONTROL_CHANGED',key)
    preserved.append(dict(target=key[0],symbol=key[1],sha256=digest))
assert len(preserved)>=36,len(preserved)
isa=json.loads(Path('/workspace/v7_coverage/p6h_decode_arch/trial03_isa.json').read_text())
depth8=next(x for x in isa if x['depth']==8)
assert depth8['depth']==8 and depth8['declared_dynamic_lds_bytes']==32768
assert depth8['opcodes'].get('buffer_load_ushort',0)==0
assert depth8['opcodes'].get('buffer_load_dwordx2',0)>0 and depth8['direct_lds_loads']>0
matches=[h for (target,name),h in old_hashes.items() if 'gfx936' in target and 'DecodeNnWaveKernelILi8E' in name]
assert matches==[depth8['sha256']]
assert depth8['metadata']['.private_segment_fixed_size']==0
source=Path('/workspace/v7_coverage/p6h_decode_arch/trial03/source')
files={}
for p in source.rglob('*'):
    if p.is_file():
        rel=p.relative_to(source);q=ROOT/'archived_source'/rel;q.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,q);files[str(rel)]=r.sha(p)
record=dict(ok=True,archived_official_benchmark=str(old),archived_binary_sha256=expected,
  saved_current_binary_sha256=r.sha(current),source_sha256=files,
  preserved_control_shaders=preserved,depth8_isa=depth8,
  protocol='Both arms use the archived official benchmark through its formal build/bin entry; original binary restored after the run',gpu_jobs=0)
(ROOT/'binary_audit.json').write_text(json.dumps(record,indent=2))
print('ARCHIVED_OFFICIAL_BINARY_PASS old_control_shaders_preserved='+str(len(preserved)),flush=True)
