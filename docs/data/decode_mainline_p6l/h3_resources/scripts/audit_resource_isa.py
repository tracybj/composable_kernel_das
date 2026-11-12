"""Read-only actual resource/schedule audit for the isolated resource build."""
from pathlib import Path
import argparse,collections,json,mmap,re,sys
import shader_audit,h3_isa_schedule as schedule
ROOT=Path(__file__).resolve().parent
ISO=ROOT/'isolated_resource_build'
sys.path.insert(0,'/workspace/v7_coverage/p6d')
import elf_isa as e

def static_schedule(item,kind,rows):
    mf=rows//(2 if rows==4 else 1) if kind=='Compact' else 1
    nf=128//(4//(2 if rows==4 else 1))//16 if kind=='Compact' else 128//(8//rows)//16
    cycle=6 if kind=='Compact' else 12
    ins=schedule.instructions(item['assembly']);regions=schedule.cyclic_regions(ins)
    loops=[g for g in regions if any(x['opcode'].startswith('v_mmac') for x in g)]
    assert len(loops)==1,(kind,rows,[(len(g),sum(x['opcode'].startswith('v_mmac') for x in g)) for g in loops])
    loop=loops[0];mmac=sum(x['opcode'].startswith('v_mmac') for x in loop)
    assert mmac==cycle*mf*nf*2,(kind,rows,mmac,cycle*mf*nf*2)
    barriers=[i for i,x in enumerate(loop) if x['opcode']=='s_barrier'];assert len(barriers)==cycle
    groups=[]
    for j,index in enumerate(barriers):
        block=loop[index+1:barriers[j+1] if j+1<len(barriers) else len(loop)]
        reads=[x for x in block if x['opcode'].startswith('ds_read')]
        calcs=[x for x in block if x['opcode'].startswith('v_mmac')]
        assert len(reads)==(mf+nf//2)*2 and len(calcs)==mf*nf*2,(kind,rows,j,len(reads),len(calcs))
        first=next(i for i,x in enumerate(block) if x['opcode'].startswith('ds_read'))
        compute=next(i for i,x in enumerate(block) if x['opcode'].startswith('v_mmac'))
        assert not any(x['opcode']=='s_waitcnt' and 'lgkmcnt' in x['operands'] for x in block[first:compute])
        written=set().union(*(schedule.vgprs(x['operands'].split(',')[0]) for x in reads));used=set()
        for calc in calcs:
            parts=calc['operands'].split(',');used|=schedule.vgprs(parts[1])|schedule.vgprs(parts[2])
        assert not written&used,(kind,rows,j,'Overwrite current operands')
        groups.append(dict(stage=j,read_registers=sorted(written),used_registers=sorted(used),read_pcs=[hex(x['address']) for x in reads],mmac_pcs=[hex(x['address']) for x in calcs]))
    for j in range(2,len(groups)):
        assert groups[j]['used_registers']==groups[j-2]['read_registers'],(kind,rows,j,'Not two stages ahead')
    return dict(kind=kind,rows=rows,cycle=cycle,main_loop=dict(start_pc=hex(loop[0]['address']),end_pc=hex(loop[-1]['address']),mmac=mmac,barriers=len(barriers)),two_stage_matches=len(groups)-2,no_immediate_lg_wait=True,groups=groups,static_only=True,not_timing_or_GPU_correctness_proof=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',default=str(ISO/'cooperative_resource_probe.o'));ap.add_argument('--tag',default='resource_probe');ap.add_argument('--full',action='store_true');args=ap.parse_args()
    binary=Path(args.binary);inventory=shader_audit.inventory(binary)
    reference=shader_audit.inventory('/workspace/composable_kernel/build/bin/tile_example_grouped_gemm_device_args_benchmark')
    old={k:v for k,v in reference.items() if args.full or 'DecodeNnCooperativeKernel' in v['symbol']}
    missing=[k for k in old if k not in inventory]
    changed=[k for k,v in old.items() if k in inventory and inventory[k]['machine_sha256']!=v['machine_sha256']]
    result=dict(binary=str(binary),old_shader_count=len(old),missing=missing,changed=changed,new=[],static_schedules=[],cpu_only=True,GPU=False)
    e.ROOT=ISO
    with binary.open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        fat=next(s for s in e.sections(data) if s['name']=='.hip_fatbin')
        for triple,code in e.bundles(data[fat['off']:fat['off']+fat['size']]):
            if code[:4]!=b'\x7fELF':continue
            sections=e.sections(code);symbols=e.symbols(code,sections);meta=e.metadata(code,sections)
            for name,fields in meta.items():
                match=re.search(r'DecodeNnCooperative(Compact|Wide)KernelILi(\d+)EE',name)
                if not match:continue
                kind,rows=match[1],int(match[2]);padded=rows*16
                assert rows in ([1,2,4] if kind=='Compact' else [2,4])
                entry=dict(target=triple,symbol=name,kind=kind,padded_M=padded,
                    vgpr=fields.get('.vgpr_count'),sgpr=fields.get('.sgpr_count'),
                    scratch=fields.get('.private_segment_fixed_size',0),
                    vgpr_spills=fields.get('.vgpr_spill_count',0),sgpr_spills=fields.get('.sgpr_spill_count',0),
                    waves_per_CTA=4 if kind=='Compact' else 8,
                    expected_dynamic_LDS_bytes=(2*rows*1024+3*8192 if kind=='Compact' else 4*(rows*1024+8192)),
                    LDS_is_source_launch_constant_not_metadata_capability=True)
                assert entry['scratch']==entry['vgpr_spills']==entry['sgpr_spills']==0
                result['new'].append(entry)
                if 'gfx936' not in triple:continue
                tag=f'{args.tag}_{kind.lower()}_m{padded}';co=ISO/'isa'/(tag+'.co');co.parent.mkdir(exist_ok=True);co.write_bytes(code)
                item=e.dump(co,code,sections,symbols,meta,name,tag)
                assert item['unknown_lines']==0
                assert item['opcodes'].get('buffer_load_ushort',0)==item['opcodes'].get('buffer_load_dwordx2',0)==0
                assert item['direct_lds_loads']>0
                item.update(kind=kind,padded_M=padded,expected_dynamic_LDS_bytes=entry['expected_dynamic_LDS_bytes'])
                (ISO/'isa'/(tag+'.json')).write_text(json.dumps(item,indent=2)+'\n')
                result['static_schedules'].append(static_schedule(item,kind,rows))
    result['counts_by_target']=dict(collections.Counter(x['target'] for x in result['new']))
    result['ok']=not missing and not changed and len(result['new'])==10 and len(result['static_schedules'])==5
    (ISO/(args.tag+'_isa_audit.json')).write_text(json.dumps(result,indent=2)+'\n')
    assert result['ok'],result
    print('RESOURCE_ACTUAL_ISA_AUDIT_PASS',len(result['new']),'new kernels,',len(result['static_schedules']),'gfx936 schedules, preserved',len(old),flush=True)
    for x in result['new']:
        print('RESOURCE_KERNEL',x['target'],x['kind'],'M',x['padded_M'],'VGPR',x['vgpr'],'LDS',x['expected_dynamic_LDS_bytes'],'scratch',x['scratch'],flush=True)

if __name__=='__main__':main()
