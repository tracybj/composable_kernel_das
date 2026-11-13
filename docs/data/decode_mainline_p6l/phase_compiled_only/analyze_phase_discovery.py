"""Audit every completed three-round pair without turning exploration into acceptance."""
from pathlib import Path
import csv,hashlib,json,math,statistics
import runtime as r
ROOT=Path(__file__).resolve().parent

def journal(path):
    if not path.exists():return []
    lines=path.read_text().splitlines(keepends=True)
    # A writer may still be appending the final line. Completed pair keys must
    # already reference flushed, newline-terminated physical records.
    return [json.loads(s) for s in lines if s.endswith('\n')]

def main():
    source=ROOT/'phase_discovery_results.json'
    if not source.exists():
        print('NO_CANDIDATE_DISCOVERY_RESULTS_YET');return
    pairs=json.loads(source.read_text());jobs=journal(ROOT/'jobs.jsonl');bykey={x['key']:x for x in jobs}
    assert len(bykey)==len(jobs)
    original_path=Path('/workspace/composable_kernel/docs/data/decode_depth8_replay_p6k/results.json')
    assert hashlib.sha256(original_path.read_bytes()).hexdigest()=='00c80c665735d59165947e5c9458e4e3cb0327df1a47dee2926fe725af4e9d3d'
    originals={x['ID']:x for x in json.loads(original_path.read_text())}
    gates={x['key']:x for x in journal(ROOT/'gates.jsonl')}
    kernels={x['key']:x for x in journal(ROOT/'kernel_checks.jsonl')}
    rows=[];grouped={};seen=set()
    for row in pairs:
        group=(row['plan_sha256'],row['binary_sha256'],row['epoch'])
        identity=(group,row['pair']);assert identity not in seen;seen.add(identity)
        records=[bykey[k] for k in row['keys']];assert len(records)==8
        assert [x['kind'] for x in records]==['accuracy','accuracy']+['perf']*6
        assert [x['variant'] for x in records]==['base','candidate','base','candidate','candidate','base','base','candidate']
        assert [x.get('round') for x in records[2:]]==[1,1,2,2,3,3]
        origin=originals[row['ID']]
        assert row['shape']=={z:origin[z] for z in ['M','N','K','G']}
        assert row['frozen_BLAS_ms']==origin['frozen_BLAS_ms']
        expected_elements=origin['M']*origin['N']*origin['G']
        arms={'base':[],'candidate':[]}
        for rec in records:
            assert rec['ok'] and rec['physical_gpu']==4 and rec['shape']==row['shape']
            gate=gates[rec['key']];assert gate['rc']==0 and gate['hcu']=='0.0%' and gate['mem_available_kib']>=64*1024*1024 and 'MKFD' not in gate['output'].upper()
            assert all(kernels[rec['key']+suffix]['rc']==0 and not kernels[rec['key']+suffix]['events'] for suffix in ['_pre','_post'])
            assert rec['environment']['HIP_VISIBLE_DEVICES']==rec['environment']['CUDA_VISIBLE_DEVICES']=='4'
            assert (rec['plan_sha256'],rec['benchmark_sha256'],rec['epoch'])==group
            assert rec['binary_sha256']==row['binary_sha256']
            assert rec['pair']==row['pair']
            arm=rec['variant'];expected=row[arm+'_instance']
            assert rec['selected']==expected
            actual=r.parse_output((Path(rec['path'])/'run.log').read_text(),rec['kind'],expected,3 if rec['kind']=='accuracy' else 1)
            if rec['kind']=='accuracy':
                assert all(int(x[1])==expected_elements and int(x[2])==0 for x in actual['comparisons'])
            else:
                assert actual['result']==rec['result'];arms[arm].append(actual['result'])
        a=statistics.median(x['median_ms'] for x in arms['base']);b=statistics.median(x['median_ms'] for x in arms['candidate'])
        assert math.isclose(a,row['base_ms'],rel_tol=1e-12) and math.isclose(b,row['candidate_ms'],rel_tol=1e-12)
        ratio=b/a;delta=100*(ratio-1);assert math.isclose(delta,row['delta_pct'],rel_tol=1e-10,abs_tol=1e-10)
        cv=any(x['cv']>.05 for x in arms['base']+arms['candidate']);assert cv==row['high_cv']
        blas=origin['frozen_BLAS_ms'];original_weak=origin['phase']=='decode' and origin['R_frozen']<.99
        out=dict(ID=row['ID'],**row['shape'],phase=origin['phase'],variant=row['variant'],control=row['control'],route_compare=row.get('route_compare',False),base_instance=row['base_instance'],candidate_instance=row['candidate_instance'],base_ms=a,candidate_ms=b,delta_pct=delta,high_cv=cv,base_round_ms=[x['median_ms'] for x in arms['base']],candidate_round_ms=[x['median_ms'] for x in arms['candidate']],base_round_cv=[x['cv'] for x in arms['base']],candidate_round_cv=[x['cv'] for x in arms['candidate']],frozen_BLAS_ms=blas,R_frozen_original=origin['R_frozen'],R_frozen_base=blas/a,R_frozen_candidate=blas/b,original_weak=original_weak,point_history_cross=original_weak and blas/b>=.99,all3_paired_cross=original_weak and all(blas/x['median_ms']<.99 for x in arms['base']) and all(blas/x['median_ms']>=.99 for x in arms['candidate']),formal_seven_block_acceptance=False,binary_sha256=group[1],epoch=group[2],plan_sha256=group[0],keys=row['keys'])
        rows.append(out);grouped.setdefault(group,[]).append(out)
    summaries=[]
    for group,rs in grouped.items():
        variants={}
        for variant in sorted({x['variant'] for x in rs}):
            cases=[x for x in rs if x['variant']==variant and not x['control']]
            if not cases:continue
            variants[variant]=dict(pairs=len(cases),improve_gt2=sum(x['delta_pct']< -2 for x in cases),point_improve_ge5=sum(x['delta_pct']<= -5 for x in cases),regress_gt2=sum(x['delta_pct']>2 for x in cases),within2=sum(abs(x['delta_pct'])<=2 for x in cases),high_cv=sum(x['high_cv'] for x in cases),geometric_candidate_over_base=math.exp(statistics.mean(math.log(x['candidate_ms']/x['base_ms']) for x in cases)),equal_case_time_sum_ratio=sum(x['candidate_ms'] for x in cases)/sum(x['base_ms'] for x in cases))
        summaries.append(dict(plan_sha256=group[0],binary_sha256=group[1],epoch=group[2],pairs=len(rs),variants=variants))
    registered=r.read(ROOT/'phase_discovery_plan.json')['discovery_pairs']
    assert len(pairs)==len(registered)==46 and {x['pair'] for x in pairs}=={x['key'] for x in registered}
    failures=[dict(key=x['key'],kind=x['kind'],rc=x['rc'],parse_error=x.get('parse_error'),path=x['path']) for x in jobs if str(x.get('purpose','')).startswith('phase_') and not x['ok']]
    report=dict(at=r.stamp(),stage='Three-round discovery only',input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed_pairs=len(rows),expected_pairs=46,complete_phase=len(rows)==46 and len(grouped)==1,groups=summaries,failed_jobs=failures,rows=rows,note='Keep all builds/epochs and losses separately. No fastest-round selection, no best-per-shape mixture, and no seven-block stability/attribution claim from these three-round data.')
    r.write(ROOT/'phase_discovery_analysis.json',report)
    fields=['ID','M','N','K','G','phase','variant','control','base_instance','candidate_instance','base_ms','candidate_ms','delta_pct','high_cv','base_round_ms','candidate_round_ms','base_round_cv','candidate_round_cv','frozen_BLAS_ms','R_frozen_original','R_frozen_base','R_frozen_candidate','point_history_cross','all3_paired_cross','formal_seven_block_acceptance','binary_sha256','epoch','plan_sha256']
    with (ROOT/'phase_discovery_analysis.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],list) else row[k] for k in fields})
    print('DISCOVERY_RAW_PAIR_AUDIT_PASS',len(rows),'expected46; confirmation_not_evaluated',flush=True)
    for x in summaries:print('DISCOVERY_GROUP',json.dumps(x),flush=True)

if __name__=='__main__':main()
