"""CPU preservation audit against this round's frozen cache binary plus new resources."""
import importlib.util,json
import runtime as r

spec=importlib.util.spec_from_file_location('p6f_isa','/workspace/v7_coverage/p6e/cache_isa.py')
isa=importlib.util.module_from_spec(spec);spec.loader.exec_module(isa)
old=isa.kernels(r.ROOT/'bench_cache')
binary=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
new=isa.kernels(binary)
preserved=[]
for name,x in old.items():
    assert name in new and x['sha256']==new[name]['sha256'],('Old shader changed',x['demangled'])
    preserved.append(dict(symbol=name,sha256=x['sha256']))
entries=[('tail128','GemmPipelineV7Tail128<',32768),('tail256','GemmPipelineV7Tail256<',49152),
         ('k32_steady','GemmPipelineV7SmallMK32Steady<',49152),
         ('k16_steady','GemmPipelineV7SmallMSteady<',32768),
         ('small_noslc','GemmPipelineV7SmallMNoSlc<',32768)]
resources=[]
for variant,needle,lds in entries:
    matches=[x for x in new.values() if needle in x['demangled']]
    assert len(matches)==1,(variant,len(matches))
    x=matches[0];md=x['metadata']
    assert md['.private_segment_fixed_size']==0,(variant,md)
    assert md.get('.vgpr_spill_count',0)==0 and md.get('.sgpr_spill_count',0)==0,(variant,md)
    resources.append(dict(variant=variant,declared_dynamic_lds=lds,**x))
# The optional cache control must isolate exactly the SLC machine-bit change.
source=[x for x in old.values() if 'GemmPipelineV7SmallM<' in x['demangled'] and 'V7SmallMGeometry<128>' in x['demangled']]
target=next(x for x in resources if x['variant']=='small_noslc')
assert len(source)==1,len(source)
isolated=source[0]['without_slc_sha256']==target['without_slc_sha256']
record=dict(at=r.stamp(),benchmark_sha256=r.sha(binary),old_cache_binary_sha256=r.sha(r.ROOT/'bench_cache'),
    preserved=preserved,new_resources=resources,small_cache_only_slc_bits_differ=isolated)
(r.ROOT/'impl_isa_audit.json').write_text(json.dumps(record,indent=2))
assert isolated,'Small cache control changed additional shader instructions; do not claim isolated cache hint'
print('IMPL_SHADER_PRESERVATION_PASS',len(preserved),'NEW_ZERO_SCRATCH_SPILL',len(resources),'SMALL_CACHE_ISOLATION',isolated)
