"""Classify the selected kernels by verified steady K-loop buffer organization."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent


def main():
    rows=json.loads((ROOT/'kernels.json').read_text())
    classes=collections.defaultdict(list)
    geometries=collections.defaultdict(list)
    kernels=[]
    for row in rows:
        lines=Path(row['assembly']).read_text().splitlines()
        labels=[(i,re.search(r'<(.*?)>:',line)[1]) for i,line in enumerate(lines)
                if re.search(r'<(.*?)>:',line)]
        start=next(i for i,name in labels if 'openLoopL' in name)
        tail=next((i for i,name in labels if 'oddexit_31' in name),None)
        if tail is None:
            tail=next(i for i,name in labels if 'LoopEndL_evenexit' in name)
        assert start<tail
        span=lines[start+1:tail]
        parsed=[re.match(r'^\s*(\w+)\s+(.*?)\s*//',line) for line in span]
        instructions=[(m[1],m[2]) for m in parsed if m]
        loads=[(op,arg) for op,arg in instructions if op.startswith('buffer_load')]
        assert loads and all(re.search(r'\blds\b',arg) for op,arg in loads)
        assert any(op=='s_barrier' for op,arg in instructions)
        descriptors=collections.Counter(re.search(r's\[\d+:\d+\]',arg)[0] for op,arg in loads)
        assert len(descriptors)==2,descriptors
        anchors=collections.defaultdict(set)
        for line in span:
            m=re.search(r's_add_u32 m0, (s\d+), (0x[0-9a-f]+|\d+)',line)
            if m:anchors[m[1]].add(int(m[2],0))
        assert len(anchors)==2
        values=[sorted(v) for v in anchors.values()]
        assert values[0]==values[1]
        offsets=values[0]
        assert len(offsets) in [2,4]
        assert offsets==[i*offsets[1] for i in range(len(offsets))]
        shift=[dict(line=i+1,source_register=m[1],shift=int(m[2]))
               for i,line in enumerate(lines[:start])
               if (m:=re.search(r'\bs_lshr_b32\s+s5,\s*(s\d+),\s*([45])\s*//',line))]
        assert shift and all(2**x['shift']==row['logical_tile'][2] for x in shift)
        depth=len(offsets);k=row['logical_tile'][2]
        tag=f'k{k}_lds{depth}'
        classes[tag].append(row)
        geometries[tuple(row['logical_tile'])].append(row)
        ops=collections.Counter(op for op,arg in instructions)
        item=dict(index=row['index'],kernel_name=row['symbol'],machine_sha256=row['sha256'],
                  ids=row['ids'],total=row['total'],decode=row['decode'],prefill=row['prefill'],
                  logical_tile=row['logical_tile'],class_tag=tag,lds_buffers=depth,
                  main_loop_lines=[start+1,tail],lds_plane_offsets=offsets,
                  k_counter_shifts=shift,steady_load_descriptors=dict(descriptors),
                  steady_vmem_loads=len(loads),steady_loads_all_direct_to_lds=True,
                  steady_cta_barriers_static=ops['s_barrier'],
                  steady_waits=dict(collections.Counter(arg for op,arg in instructions if op=='s_waitcnt')),
                  threads=row['metadata']['.max_flat_workgroup_size'],
                  lds_bytes=row['metadata']['.group_segment_fixed_size'],
                  vgpr=row['metadata']['.vgpr_count'],sgpr=row['metadata']['.sgpr_count'],
                  scratch_bytes=row['metadata'].get('.private_segment_fixed_size',0),
                  raw_assembly=row['assembly'])
        if depth==2:
            assert row['index']==12 and 'PGR1_' in row['symbol']
            assert item['steady_waits'].get('vmcnt(0)',0)>0
            assert item['steady_waits'].get('lgkmcnt(0)',0)>0
        else:
            assert any(re.fullmatch(r'lgkmcnt\([1-9]\d*\)',x) for x in item['steady_waits'])
        kernels.append(item)
    assert set(classes)=={'k16_lds4','k32_lds4','k16_lds2'}
    def counts(group):
        return dict(kernel_instances=len(group),total=sum(x['total'] for x in group),
                    decode=sum(x['decode'] for x in group),prefill=sum(x['prefill'] for x in group))
    summary=dict(scope='Frozen1193 BF16 NN GroupedGemm on gfx936, not all BLASLt kernels',
                 source='1178 historical kernel_name mappings plus 15 frozen added-shape mappings',
                 gpu_jobs=0,optimization_state='paused',code_object_sha256=json.loads((ROOT/'dependencies.json').read_text())[str(Path('/workspace/Grouped_Geem/hipblaslt-install/lib/hipblaslt/library_gfx936/TensileLibrary_gfx936.co'))]['sha256'],
                 total=counts(rows),architectural_frameworks=1,
                 architectural_framework='CTA-cooperative A/B direct-to-LDS, LDS-to-operand-register, MMAC, CTA synchronization',
                 buffer_organization_families=2,
                 class_definition='Verified K block size combined with physical LDS ring organization; groups may contain different operand/wave/wait schedules',
                 classes={tag:counts(group) for tag,group in sorted(classes.items())},
                 logical_geometries=[dict(logical_tile=list(key),**counts(group)) for key,group in sorted(geometries.items())],
                 limitations=['Steady K-loop classification, not equality of complete instruction schedules or edge/epilogue paths',
                              'Historical selection map; no new GPU selection or performance reproduction',
                              'Do not infer BLASLt library-wide counts, other dtypes, layouts or architectures',
                              'Four ring planes do not imply exactly four total stages of lookahead including operand preloads'])
    assert summary['total']==dict(kernel_instances=14,total=1193,decode=881,prefill=312)
    assert summary['classes']['k16_lds4']==dict(kernel_instances=8,total=957,decode=645,prefill=312)
    assert summary['classes']['k32_lds4']==dict(kernel_instances=5,total=221,decode=221,prefill=0)
    assert summary['classes']['k16_lds2']==dict(kernel_instances=1,total=15,decode=15,prefill=0)
    (ROOT/'pipeline_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (ROOT/'pipeline_kernels.json').write_text(json.dumps(kernels,ensure_ascii=False,indent=2))
    fields=['index','class_tag','kernel_name','total','decode','prefill','logical_tile','lds_buffers','threads','lds_bytes','vgpr','sgpr','machine_sha256']
    with (ROOT/'pipeline_kernels.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in kernels:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in fields})
    print('PIPELINE_CLASSIFICATION_PASS',json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
