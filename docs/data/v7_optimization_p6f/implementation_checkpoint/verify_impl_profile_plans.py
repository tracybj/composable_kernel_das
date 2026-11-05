"""CPU symbol audit before any new profile command; freeze corrected control geometry."""
import importlib.util,json,mmap,subprocess
from pathlib import Path
import runtime as r
spec=importlib.util.spec_from_file_location('p6f_elf','/workspace/v7_coverage/p6d/elf_isa.py')
elf=importlib.util.module_from_spec(spec);spec.loader.exec_module(elf)
binary=r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'
names=[]
with binary.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as host:
    fat=next(s for s in elf.sections(host) if s['name']=='.hip_fatbin')
    for triple,data in elf.bundles(host[fat['off']:fat['off']+fat['size']]):
        if 'gfx936' not in triple or data[:4]!=b'\x7fELF':continue
        names.extend(name for name in elf.symbols(data,elf.sections(data)) if 'kentry' in name)
demangled=subprocess.check_output(['/opt/dtk/aillvm/bin/llvm-cxxfilt'],input='\n'.join(names),text=True).splitlines()
assert len(names)==len(demangled)
record=[]
for source,target in [('tail_profiles_frozen.json','tail_profiles_verified_frozen.json'),
                      ('decode_profiles_frozen.json','decode_profiles_verified_frozen.json')]:
    plan=json.loads((r.ROOT/source).read_text())
    for job in plan['jobs']:
        if job['variant']=='k16_steady':
            job['kernel_filter']='GemmPipelineV7SmallMSteady'
        if job['variant']=='tail_control':
            job['logical_tile']=[64,128,64]
            job['kernel_filter']='GemmPipelineAgBgCrCompV4'
        matches=[dict(symbol=n,demangled=d) for n,d in zip(names,demangled) if job['kernel_filter'] in d]
        assert matches,(job['key'],job['kernel_filter'])
        record.append(dict(key=job['key'],filter=job['kernel_filter'],matches=matches))
    path=r.ROOT/target;assert not path.exists()
    plan.update(at=r.stamp(),benchmark_sha256=r.sha(binary),note='Kernel filters matched actual final gfx936 symbols;4003 logical64x128x64 corrected from prior frozen placeholder before first GPU profile')
    path.write_text(json.dumps(plan,indent=2))
(r.ROOT/'impl_profile_symbol_audit.json').write_text(json.dumps(dict(benchmark_sha256=r.sha(binary),checks=record),indent=2))
print('IMPL_PROFILE_SYMBOL_AUDIT_PASS',len(record))
