"""Actual gfx936 CFG/register proof for the chosen compute-before-ready code."""
from pathlib import Path
import hashlib,json,mmap,re,sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT));import h3_isa_schedule as schedule
sys.path.insert(0,'/workspace/v7_coverage/p6d');import elf_isa as e

def audit(item,rows,wide):
    mf=1 if wide else rows//(2 if rows==4 else 1)
    nf=128//((8//rows) if wide else (4//(2 if rows==4 else 1)))//16
    cycle=12 if wide else 6
    ins=schedule.instructions(item['assembly']);regions=schedule.cyclic_regions(ins)
    loops=[g for g in regions if any(x['opcode'].startswith('v_mmac') for x in g)]
    assert len(loops)==1,(rows,wide,'cyclic regions',len(loops))
    loop=loops[0];barriers=[i for i,x in enumerate(loop) if x['opcode']=='s_barrier']
    assert len(barriers)==cycle
    assert sum(x['opcode'].startswith('v_mmac') for x in loop)==cycle*mf*nf*2
    # Rotate an actual cyclic region: MMAC0 precedes barrier0 in this code.
    # After barrier j we read operands j+2, then compute j+1; the latter's
    # operands were read in the previous segment and drained by ready j.
    rotated=loop[barriers[0]:]+loop[:barriers[0]]
    cuts=[i for i,x in enumerate(rotated) if x['opcode']=='s_barrier'];groups=[]
    for j,cut in enumerate(cuts):
        segment=rotated[cut+1:cuts[j+1] if j+1<len(cuts) else len(rotated)]
        reads=[x for x in segment if x['opcode'].startswith('ds_read')]
        calcs=[x for x in segment if x['opcode'].startswith('v_mmac')]
        assert len(reads)==(mf+nf//2)*2 and len(calcs)==mf*nf*2,(rows,wide,j,len(reads),len(calcs))
        first=next(i for i,x in enumerate(segment) if x['opcode'].startswith('ds_read'))
        compute=next(i for i,x in enumerate(segment) if x['opcode'].startswith('v_mmac'))
        assert not any(x['opcode']=='s_waitcnt' and 'lgkmcnt' in x['operands'] for x in segment[first:compute]),(rows,wide,j,'Future read immediately waited before MMAC')
        writes=set().union(*(schedule.vgprs(x['operands'].split(',')[0]) for x in reads));uses=set()
        for x in calcs:
            args=x['operands'].split(',');uses|=schedule.vgprs(args[1])|schedule.vgprs(args[2])
        assert not writes&uses,(rows,wide,j,'Future-read/current-MMAC alias')
        after=next(i for i,x in enumerate(segment) if x['opcode'].startswith('v_mmac'))
        assert any(x['opcode']=='s_waitcnt' and 'lgkmcnt(0)' in x['operands'] for x in segment[after:]),(rows,wide,j,'Missing unchanged drain before next barrier')
        groups.append(dict(segment=j,reads=sorted(writes),uses=sorted(uses),read_pcs=[hex(x['address']) for x in reads],mmac_pcs=[hex(x['address']) for x in calcs]))
    for j in range(1,len(groups)):
        assert groups[j]['uses']==groups[j-1]['reads'],(rows,wide,j,'Wrong ready operand set')
    anchors=[dict(line=x['line'],text=x['text']) for x in loop if x['opcode'].startswith(('s_waitcnt','s_barrier','buffer_load','ds_read','v_mmac'))][:32]
    return dict(rows=rows,wide=wide,cycle=cycle,main_loop=dict(start_pc=hex(loop[0]['address']),end_pc=hex(loop[-1]['address']),line_start=loop[0]['line'],line_end=loop[-1]['line']),segments=groups,instruction_anchors=anchors,operand_matches=len(groups)-1,unchanged_LG0_drain=True,not_a_GPU_timing_or_precision_proof=True)

def main():
    binary=HERE/'phase_probe.o';e.ROOT=HERE;proofs=[]
    with binary.open('rb') as file,mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as data:
        fat=next(x for x in e.sections(data) if x['name']=='.hip_fatbin')
        for target,code in e.bundles(data[fat['off']:fat['off']+fat['size']]):
            if 'gfx936' not in target or code[:4]!=b'\x7fELF':continue
            sec=e.sections(code);symbols=e.symbols(code,sec);meta=e.metadata(code,sec)
            for name,fields in meta.items():
                m=re.search(r'DecodeNnPhaseProbeKernelILi(\d+)ELb([01])ELb1EE',name)
                if not m:continue
                rows,wide=map(int,m.groups());tag=f'phase_probe_first_m{rows*16}_w{8 if wide else 4}'
                co=HERE/'isa'/(tag+'.co');co.parent.mkdir(exist_ok=True);co.write_bytes(code)
                item=e.dump(co,code,sec,symbols,meta,name,tag);assert item['unknown_lines']==0
                proof=audit(item,rows,bool(wide));proof.update(symbol=name,machine_sha256=item['sha256'],VGPR=fields['.vgpr_count']);proofs.append(proof)
    assert len(proofs)==5
    result=dict(ok=True,cpu_only=True,GPU=False,source_sha256=hashlib.sha256((HERE/'phase_probe.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),proofs=proofs)
    (HERE/'probe_schedule_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ACTUAL_PHASE_PROBE_CFG_REGISTER_PASS',len(proofs),'selected kernels; no GPU',flush=True)

if __name__=='__main__':main()
