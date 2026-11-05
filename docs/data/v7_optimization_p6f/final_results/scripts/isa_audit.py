"""CPU-only byte preservation and M01=5 cache modifier isolation."""
import importlib.util,json,re
from pathlib import Path
import runtime as r
spec=importlib.util.spec_from_file_location('p6f_kernel_bytes','/workspace/v7_coverage/p6e/cache_isa.py');isa=importlib.util.module_from_spec(spec);spec.loader.exec_module(isa)
before=isa.kernels(Path('/workspace/v7_coverage/p6e/bench_small'))
after=isa.kernels(r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark')
preserved=[]
for name,row in before.items():
    assert name in after,name
    assert row['sha256']==after[name]['sha256'],('Existing shader changed',row['demangled'])
    preserved.append(dict(symbol=name,sha256=row['sha256']))
pairs=[]
for walk in [1,5,8]:
    matched=[]
    for row in after.values():
        prefix=row['demangled'].split('GemmPipelineV7')[0]
        m=re.search(r'false, false>, 1, (\d+)>',prefix)
        if m and int(m[1])==walk:
            if 'GemmPipelineV7Uniform<' in row['demangled'] and '(ck_tile::V7UniformKind)0' in row['demangled']:matched.append(('slc',row))
            elif 'GemmPipelineV7CacheNoSlc<' in row['demangled']:matched.append(('noslc',row))
    data=dict(matched);assert set(data)=={'slc','noslc'},(walk,set(data))
    isolated=data['slc']['without_slc_sha256']==data['noslc']['without_slc_sha256']
    pairs.append(dict(m01=walk,only_slc_machine_bits_differ=isolated,variants=data))
    assert isolated,('Cache factor not isolated at ISA level',walk)
out=dict(at=r.stamp(),benchmark_sha256=r.sha(r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'),original_shaders_unchanged=preserved,cache_pairs=pairs)
(r.ROOT/'cache_isa_audit.json').write_text(json.dumps(out,indent=2))
print('CACHE_SHADER_BYTE_PRESERVATION_PASS',len(preserved))
print('CACHE_SLC_MACHINE_BIT_ISOLATION_PASS walks=1,5,8')
