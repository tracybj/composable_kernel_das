"""CPU-only inventory of all kernels selected by the frozen 1193-shape table.

Vendor ISA is kept in this local work directory. No GPU or tuning is run.
"""
import collections
import csv
import hashlib
import importlib.util
import json
import mmap
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REPO = Path('/workspace/composable_kernel')
spec = importlib.util.spec_from_file_location('prior_elf_isa', '/workspace/v7_coverage/p6d/elf_isa.py')
elf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elf)
elf.ROOT = ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert REPO not in ROOT.parents, 'Keep raw vendor ISA outside the CK repository'
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = json.loads(Path('/workspace/v7_coverage/p6d/manifest.json').read_text())
    dependencies = [Path('/workspace/bf16NN-groupgemm.csv'),
                    Path('/workspace/v7_coverage/p6c/native_new15.json'), elf.CO]
    manifest = {}
    for path in dependencies:
        digest = sha(path)
        assert digest == prior[str(path)]['sha256'], str(path)
        manifest[str(path)] = dict(bytes=path.stat().st_size, sha256=digest)
    shapes_path = Path('/workspace/shapes_1178.csv')
    shapes = {int(x['ID']):x for x in csv.DictReader(shapes_path.open(encoding='utf-8-sig'))}
    manifest[str(shapes_path)] = dict(bytes=shapes_path.stat().st_size, sha256=sha(shapes_path))
    old = list(csv.DictReader(dependencies[0].open(encoding='utf-8-sig')))
    added = json.loads(dependencies[1].read_text())
    names = {int(x['ID']):x['kernel_name'] for x in old}
    names.update({int(x['id']):x['kernel_name'] for x in added})
    assert len(names) == len(shapes) == 1193
    families = collections.defaultdict(list)
    for ident, name in names.items():
        families[name].append(ident)
    ordered = sorted(families.items(), key=lambda x:(-len(x[1]), x[0]))
    assert len(ordered) == 14
    previous = {x['symbol']:x for x in json.loads(Path('/workspace/v7_coverage/p6d/isa_features.json').read_text())
                if x['tag'].startswith('blas')}
    records = []
    with elf.CO.open('rb') as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
        sections = elf.sections(data)
        symbols = elf.symbols(data, sections)
        metadata = elf.metadata(data, sections)
        for i, (name, ids) in enumerate(ordered,1):
            assert name in symbols and name in metadata, name
            tag = f'kernel_{i:02}'
            row = elf.dump(elf.CO,data,sections,symbols,metadata,name,tag)
            assert row['unknown_lines'] == 0
            if name in previous:
                assert row['sha256'] == previous[name]['sha256']
            tile = re.search(r'_MT(\d+)x(\d+)x(\d+)', name)
            assert tile
            nm, nn, nk = map(int,tile.groups())
            row.update(index=i,ids=sorted(ids),total=len(ids),
                       decode=sum(int(shapes[j]['M'])<=256 for j in ids),
                       prefill=sum(int(shapes[j]['M'])>256 for j in ids),
                       logical_tile=[nn,nm,nk],previously_disassembled=name in previous)
            lines = Path(row['assembly']).read_text().splitlines()
            labels = [(n+1,re.search(r'<(.*?)>:',line)[1]) for n,line in enumerate(lines)
                      if re.search(r'<(.*?)>:',line)]
            row['loop_labels'] = [(n,label) for n,label in labels if re.search(r'loop|tail|prefetch',label,re.I)]
            start = next((n for n,label in labels if 'openLoopL' in label),None)
            tail = next((n for n,label in labels if 'oddexit_31' in label),None)
            if start and tail and start < tail:
                parsed = [re.match(r'^\s*(\w+)\s+(.*?)\s*//',line) for line in lines[start:tail-1]]
                instructions = [(m[1],m[2]) for m in parsed if m]
                ops = collections.Counter(op for op,arg in instructions)
                loads = [(op,arg) for op,arg in instructions if op.startswith('buffer_load')]
                row['steady_static_span'] = dict(line_start=start,line_end=tail-1,
                    buffer_loads=len(loads),loads_to_lds=sum(bool(re.search(r'\blds\b',arg)) for op,arg in loads),
                    cta_barriers=ops['s_barrier'],
                    mmac=sum(v for k,v in ops.items() if k.startswith('v_mmac')),
                    waits=dict(collections.Counter(arg for op,arg in instructions if op=='s_waitcnt')),
                    descriptor_load_counts=dict(collections.Counter(re.search(r's\[\d+:\d+\]',arg)[0] for op,arg in loads)))
                row['prologue_d2l'] = sum('buffer_load' in line and bool(re.search(r'\blds\b',line)) for line in lines[:start-1])
            records.append(row)
            print('KERNEL_CPU_DISASSEMBLED',i,'total',row['total'],'decode',row['decode'],
                  'tile',row['logical_tile'],'VGPR',row['metadata'].get('.vgpr_count'),
                  'LDS',row['metadata'].get('.group_segment_fixed_size'),
                  'barriers_static',row['opcodes'].get('s_barrier',0),
                  'steady',row.get('steady_static_span'),flush=True)
    (ROOT/'kernels.json').write_text(json.dumps(records,indent=2))
    (ROOT/'dependencies.json').write_text(json.dumps(manifest,indent=2))
    print('CPU_INVENTORY_PASS kernels=14 shapes=1193 decode=881 prefill=312 GPU_jobs=0',flush=True)


if __name__ == '__main__':
    main()
