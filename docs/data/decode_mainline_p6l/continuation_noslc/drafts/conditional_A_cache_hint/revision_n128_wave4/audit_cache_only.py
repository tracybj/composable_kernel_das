"""Prove the probe changes only the explicit A-load SLC modifier."""
from pathlib import Path
import hashlib,json,mmap,re,sys
D=Path(__file__).resolve().parent;ROOT=D.parents[2]
sys.path.insert(0,str(ROOT))
import shader_audit,h3_isa_schedule as schedule
sys.path.insert(0,'/workspace/v7_coverage/p6d');import elf_isa as e
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,tag):
    out={};folder=D/'cache_isa'/tag;folder.mkdir(parents=True,exist_ok=False);e.ROOT=folder
    with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
        fat=next(x for x in e.sections(data) if x['name']=='.hip_fatbin')
        for target,code in e.bundles(data[fat['off']:fat['off']+fat['size']]):
            if 'gfx936' not in target or code[:4]!=b'\x7fELF':continue
            sections=e.sections(code);symbols=e.symbols(code,sections);meta=e.metadata(code,sections)
            for name,fields in meta.items():
                match=re.search(r'DecodeNnLowLdsWave4ProbeKernelILi(\d+)EE',name)
                if not match:continue
                rows=int(match[1]);co=folder/f'm{rows*16}.co';co.write_bytes(code)
                item=e.dump(co,code,sections,symbols,meta,name,f'm{rows*16}')
                assert item['unknown_lines']==0
                out[rows]=schedule.instructions(item['assembly'])
    return out
def norm(s):return ' '.join(s.split())
def main():
    old=ROOT/'drafts/conditional_n128_wave4/probe.o';new=D/'probe.o'
    before=shader_audit.inventory(old);after=shader_audit.inventory(new)
    assert set(before)==set(after) and len(before)==6
    assert all(before[k]['metadata']==after[k]['metadata'] for k in before)
    assert all(before[k]['bytes']==after[k]['bytes'] for k in before)
    a=dump(old,'slc');b=dump(new,'no_slc');assert set(a)==set(b)=={1,2,4}
    proofs=[]
    for rows in [1,2,4]:
        assert len(a[rows])==len(b[rows]);changed=[]
        for x,y in zip(a[rows],b[rows]):
            assert x['opcode']==y['opcode']
            if norm(x['operands'])==norm(y['operands']):continue
            assert x['opcode']=='buffer_load_dwordx4' and all(z in x['operands'] for z in ['idxen','offen','slc','lds'])
            assert 'slc' not in y['operands']
            assert norm(re.sub(r'\bslc\b','',x['operands']))==norm(y['operands'])
            changed.append(dict(before=x['text'],after=y['text']))
        assert changed
        proofs.append(dict(padded_M=rows*16,instructions=len(a[rows]),only_A_cache_modifier_changes=len(changed),changes=changed))
    report=dict(ok=True,CPU_only=True,GPU=False,old_object_sha256=sha(old),new_object_sha256=sha(new),
                dual_target_metadata_and_size_identical=6,gfx936_instruction_comparisons=proofs,
                mathematical_address_schedule_unchanged=True,performance_unverified=True,
                no_claim_about_undocumented_cache_hierarchy=True)
    (D/'cache_only_isa_proof.json').write_text(json.dumps(report,indent=2)+'\n')
    print('A_CACHE_ONLY_ISA_PROOF_PASS6resources/3gfx936bodies',flush=True)
if __name__=='__main__':main()
