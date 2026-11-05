"""Adding a host timing option must preserve every device shader byte."""
import importlib.util,json
import runtime as r

spec=importlib.util.spec_from_file_location('p6f_shader_reader','/workspace/v7_coverage/p6e/cache_isa.py')
isa=importlib.util.module_from_spec(spec);spec.loader.exec_module(isa)
before=isa.kernels(r.ROOT/'bench_impl_initial')
binary=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
after=isa.kernels(binary)
assert set(before)==set(after) and len(before)==18
for key,x in before.items():assert x['sha256']==after[key]['sha256'],x['demangled']
record=dict(at=r.stamp(),all_pass=True,device_shaders_byte_identical=len(before),
    before_binary_sha256=r.sha(r.ROOT/'bench_impl_initial'),after_binary_sha256=r.sha(binary),
    shader_hashes={key:x['sha256'] for key,x in after.items()},
    scope='Only host benchmark launch-count diagnostic changed; formal default remains one; no device kernel modifications')
(r.ROOT/'benchmark_only_shader_preservation.json').write_text(json.dumps(record,indent=2))
print('BENCHMARK_ONLY_ALL18_SHADER_BYTES_PRESERVED')
