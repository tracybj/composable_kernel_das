"""Audit real selected phase kernels and preserve every previous shader."""
from pathlib import Path
import json,mmap,re,sys
import plan,runtime as r,shader_audit,h3_isa_schedule as schedule
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'drafts/resource_interleave'))
from audit_phase_probe import audit as schedule_proof
sys.path.insert(0,'/workspace/v7_coverage/p6d')
import elf_isa as e

def main():
    folder=ROOT/'compile/build9';stages=r.read(folder/'results.json')
    assert len(stages)==2 and all(x['rc']==0 for x in stages)
    old=r.read(folder/'reference_shaders.json');new=shader_audit.inventory(plan.BENCH)
    missing=[k for k in old['kernels'] if k not in new]
    changed=[k for k,v in old['kernels'].items() if k in new and new[k]['machine_sha256']!=v['machine_sha256']]
    assert len(old['kernels'])==1710 and not missing and not changed
    pins=r.read(folder/'source_sha256.json');assert all(plan.sha(plan.REPO/n)==sha for n,sha in pins.items())
    kernels=[];proofs=[];e.ROOT=folder
    with plan.BENCH.open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        fat=next(x for x in e.sections(data) if x['name']=='.hip_fatbin')
        for target,code in e.bundles(data[fat['off']:fat['off']+fat['size']]):
            if code[:4]!=b'\x7fELF':continue
            sec=e.sections(code);symbols=e.symbols(code,sec);meta=e.metadata(code,sec)
            for name,fields in meta.items():
                match=re.search(r'DecodeNnCooperative(Compact|Wide)PhaseKernelILi(\d+)EE',name)
                if not match:continue
                kind,rows=match[1],int(match[2]);wide=kind=='Wide'
                item=dict(target=target,symbol=name,kind=kind,padded_M=rows*16,VGPR=fields['.vgpr_count'],SGPR=fields['.sgpr_count'],scratch=fields.get('.private_segment_fixed_size',0),VGPR_spills=fields.get('.vgpr_spill_count',0),SGPR_spills=fields.get('.sgpr_spill_count',0),waves_per_CTA=8 if wide else 4,expected_dynamic_LDS_bytes=4*(rows*1024+8192) if wide else 2*rows*1024+3*8192)
                assert item['scratch']==item['VGPR_spills']==item['SGPR_spills']==0
                kernels.append(item)
                if 'gfx936' not in target:continue
                tag=f'phase_actual_{kind.lower()}_m{rows*16}';co=folder/'isa'/(tag+'.co');co.parent.mkdir(exist_ok=True);co.write_bytes(code)
                captured=e.dump(co,code,sec,symbols,meta,name,tag);assert captured['unknown_lines']==0
                proof=schedule_proof(captured,rows,wide);proof.update(symbol=name,machine_sha256=captured['sha256'],VGPR=item['VGPR']);proofs.append(proof)
    assert len(kernels)==10 and len(proofs)==5
    result=dict(at=r.stamp(),ok=True,cpu_only=True,GPU=False,old_shader_count=1710,missing=missing,changed=changed,benchmark_sha256=plan.sha(plan.BENCH),kernels=kernels,proofs=proofs,not_timing_or_GPU_precision_evidence=True)
    r.write(ROOT/'phase_build_isa_audit.json',result)
    print('PHASE_ACTUAL_BINARY_CFG_AUDIT_PASS old1710 preserved/new10/gfx9365',flush=True)
    for x in kernels:print('PHASE_ACTUAL_RESOURCE',x['target'],x['kind'],x['padded_M'],x['VGPR'],x['expected_dynamic_LDS_bytes'],flush=True)

if __name__=='__main__':main()
