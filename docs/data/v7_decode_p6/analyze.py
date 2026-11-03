#!/usr/bin/env python3
"""Extract every dispatch, verify padded work, and aggregate independent rounds."""
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/workspace/v7_coverage/p4a')
from perfdump import dispatches

CASES = {76: (1, 2048, 768, 4), 681: (1, 1536, 2048, 4),
         633: (4, 1024, 2048, 8), 358: (8, 6144, 2048, 32)}
BASELINE = Path('/workspace/Grouped_Geem/blaslt_tuning_tools/results/bf16_nn_compare_20261027_191315/groupgemm/results.csv')
CU, SIMD, CLOCK = 80, 4, 1.6e9


def divide(a, b):
    return a/b if b else None


def metrics(c, shape, tile):
    m, n, k, g = shape
    tm, tn, tk = tile
    ctas = math.ceil(m/tm)*math.ceil(n/tn)*g
    padded_flops = 2*ctas*tm*tn*math.ceil(k/tk)*tk
    gui = c['GRBM_GUI_ACTIVE']
    ms = gui/CLOCK*1000
    rd = 32*c['TCC_EA_RDREQ_32B_sum'] + 64*(c['TCC_EA_RDREQ_sum']-c['TCC_EA_RDREQ_32B_sum'])
    wr = 32*(c['TCC_EA_WRREQ_sum']-c['TCC_EA_WRREQ_64B_sum'])+64*c['TCC_EA_WRREQ_64B_sum']
    busy = c['SQ_BUSY_CU_CYCLES']/(gui*CU)
    mmop = c['SQ_ACTIVE_INST_MMOP']/(gui*CU*SIMD)
    result = dict(ctas=ctas, gui_cycles=gui, ms_from_gui=ms,
        busy_cu_frac=busy, mmop_active_per_simd_cycle=mmop,
        mmop_active_per_busy_simd_cycle=divide(mmop,busy),
        mmop_insts=c['SQ_INSTS_MMOP'], mmop_insts_expected_padded=padded_flops/8192,
        mmop_count_matches_padded=c['SQ_INSTS_MMOP'] == padded_flops/8192,
        mmop_cycles_per_inst=divide(c['SQ_ACTIVE_INST_MMOP'],c['SQ_INSTS_MMOP']),
        waves=c['SQ_WAVES'], waves_per_cta=divide(c['SQ_WAVES'],ctas),
        mean_resident_waves_per_cu=c['SQ_LEVEL_WAVES_accum']/(gui*CU),
        mean_wave_lifetime_cycles=divide(c['SQ_WAVE_CYCLES'],c['SQ_WAVES']),
        wait_any_frac=divide(c['SQ_WAIT_INST_ANY'],c['SQ_WAVE_CYCLES']),
        l2_hit_pct=100*divide(c['TCC_HIT_sum'],c['TCC_HIT_sum']+c['TCC_MISS_sum']),
        tcp_tcc_read_latency=divide(c['TCP_TCC_READ_REQ_LATENCY_sum'],c['TCP_TCC_READ_REQ_sum']),
        tcp_read_requests=c['TCP_TCC_READ_REQ_sum'], hbm_read_MB=rd/1e6, hbm_write_MB=wr/1e6,
        hbm_GBps=(rd+wr)/ms/1e6,
        dram_credit_stall_frac=c['TCC_EA_RDREQ_DRAM_CREDIT_STALL_sum']/(gui*16),
        tcp_alloc_stall_frac=c['TCP_ALLOC_STALL_CYCLES_sum']/(gui*CU),
        tcp_pending_stall_frac=c['TCP_PENDING_STALL_CYCLES_sum']/(gui*CU),
        tcp_alloc_stall_per_busy_cu=divide(c['TCP_ALLOC_STALL_CYCLES_sum'],c['SQ_BUSY_CU_CYCLES']),
        tcp_pending_stall_per_busy_cu=divide(c['TCP_PENDING_STALL_CYCLES_sum'],c['SQ_BUSY_CU_CYCLES']),
        lds_bank_conflict_frac=divide(c['SQ_LDS_BANK_CONFLICT'],c['SQ_LDS_IDX_ACTIVE']))
    for counter in ['SQ_INSTS_SALU','SQ_INSTS_SMEM','SQ_INSTS_VALU','SQ_INSTS_VMEM',
                    'SQ_INSTS_LDS','SQ_INSTS_BRANCH','SQ_INSTS_MMOP']:
        if counter in c:
            result[counter.lower()+'_per_wave'] = divide(c[counter],c['SQ_WAVES'])
    if 'SQ_INSTS_VALU' in c:
        result['valu_minus_mmop_insts_per_wave'] = divide(c['SQ_INSTS_VALU']-c['SQ_INSTS_MMOP'],c['SQ_WAVES'])
        result['valu_minus_mmop_count_fraction'] = divide(c['SQ_INSTS_VALU']-c['SQ_INSTS_MMOP'],c['SQ_INSTS_VALU'])
    if tile == (256,256,16) and 'SQ_INSTS_VMEM' in c:
        result['vmem_above_two_loads_per_kstage_per_wave'] = divide(c['SQ_INSTS_VMEM'],c['SQ_WAVES'])-2*(k/16)
    return result


def main():
    with BASELINE.open(encoding='utf-8-sig') as f:
        frozen = {int(r['ID']): r for r in csv.DictReader(f)}
    jobs = [json.loads(line) for line in (ROOT/'jobs.jsonl').read_text().splitlines()]
    rows, raw_rows = [], []
    good = {r['key']: r for r in jobs if r['ok']}
    for r in good.values():
        case_id, backend = r['id'], r['backend']
        if r['suite'] == 'timing':
            rows.append(dict(r['result'], id=case_id, backend=backend, suite='timing', round=r['round']))
            continue
        if backend == 'ck':
            tile = (256,256,16)
        else:
            # Native GroupedGemm presents row-major C=A@B as column-major
            # C^T=B^T@A^T: its m=N and n=M (bridge.cpp:132-133).
            native_tile = tuple(map(int,re.search(r'MT(\d+)x(\d+)x(\d+)',frozen[case_id]['kernel_name']).groups()))
            tile = (native_tile[1], native_tile[0], native_tile[2])
        assert len(r['perf_files']) == 1, r
        ds = dispatches(r['perf_files'][0])
        assert ds, r
        raw_rows.append(dict(key=r['key'], path=r['perf_files'][0], dispatch_count=len(ds), dispatches=ds))
        c = ds[-1]
        if r['suite'] == 'gc':
            log = (Path(r['path'])/'run.log').read_text()
            dispatch_ns = [int(x) for x in re.findall(r'dispatch kernel .*?execution\((\d+) ns\)',log)]
            replay_ns = [int(x) for x in re.findall(r'replay kernel .*?execution\((\d+) ns\)',log)]
            gpu_times = dict(aql_last_dispatch_ms=dispatch_ns[-1]/1e6 if dispatch_ns else None,
                             aql_dispatch_ms=[x/1e6 for x in dispatch_ns],
                             aql_replay_ms=[x/1e6 for x in replay_ns],
                             aql_replay_median_ms=statistics.median(replay_ns)/1e6 if replay_ns else None)
            rows.append(dict(id=case_id, backend=backend, suite='gc', round=r['round'], tile=tile,
                dispatch_count=len(ds), **gpu_times, **metrics(c,CASES[case_id],tile)))
        else:
            hit = sum(v for k,v in c.items() if k.startswith('UTCL2') and k.endswith('HIT'))
            miss = sum(v for k,v in c.items() if k.startswith('UTCL2') and k.endswith('MISS'))
            ph = [v for k,v in c.items() if k.startswith('UMC_PAGE_HIT')]
            rows.append(dict(id=case_id, backend=backend, suite='tlb', round=r['round'],
                             dispatch_count=len(ds), utcl2_hit=hit, utcl2_miss=miss,
                             umc_page_hit_mean=statistics.mean(ph) if ph else None))
    (ROOT/'raw_counters.json').write_text(json.dumps(raw_rows,indent=2))
    (ROOT/'derived.json').write_text(json.dumps(rows,indent=2))
    aggregated = []
    for case_id in CASES:
        for backend in ['ck','blas']:
            row = dict(id=case_id, backend=backend, shape=CASES[case_id])
            for suite in ['timing','gc','tlb']:
                group = [r for r in rows if r['id'] == case_id and r['backend'] == backend and r['suite'] == suite]
                agg = dict(rounds=len(group))
                keys = sorted(set().union(*(r.keys() for r in group)))
                for key in keys:
                    values = [r[key] for r in group if isinstance(r.get(key),(int,float)) and not isinstance(r[key],bool)]
                    if key not in ['id','round'] and values:
                        agg[key] = statistics.median(values)
                        if len(values) > 1:
                            agg[key+'_range'] = [min(values),max(values)]
                if suite == 'gc':
                    agg['all_mmop_count_matches_padded'] = all(r['mmop_count_matches_padded'] for r in group)
                row[suite] = agg
            row['frozen_blas_card0_ms'] = float(frozen[case_id]['BLAS ms'])
            aggregated.append(row)
    (ROOT/'summary.json').write_text(json.dumps(aggregated,indent=2))
    fields=['id','backend','timing_ms','timing_cv','prof_ms','CU_busy','waves','resident_waves/CU','MMOP_active','L1_L2_cycles','credit_stall','HBM_MB','HBM_GB/s','padded_MMOP']
    print('\t'.join(fields))
    for r in aggregated:
        t,c=r['timing'],r['gc']
        values=[r['id'],r['backend'],t.get('median_ms'),t.get('cv'),c.get('ms_from_gui'),c.get('busy_cu_frac'),
                c.get('waves'),c.get('mean_resident_waves_per_cu'),c.get('mmop_active_per_simd_cycle'),
                c.get('tcp_tcc_read_latency'),c.get('dram_credit_stall_frac'),c.get('hbm_read_MB'),
                c.get('hbm_GBps'),c.get('mmop_insts')]
        print('\t'.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in values))


if __name__ == '__main__':
    main()
