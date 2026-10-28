#!/usr/bin/env python3
"""Audit a completed one-round public-API comparison against all raw shapes."""
from pathlib import Path
import argparse,csv,hashlib,json,math,statistics,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('out');a=ap.parse_args()
    out=Path(a.out);base=Path('/workspace/GroupGemm/baseline/int8_nt')
    sys.path.insert(0,str(base));from bench_int8_nt import parse_shapes
    shapes=parse_shapes(base.parent/'moe_grouped_gemm_tuning_shape_final.md')
    records=[json.loads(l) for l in (out/'records.jsonl').read_text().splitlines()]
    rows=list(csv.DictReader((out/'results.csv').open(encoding='utf-8-sig')))
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['mode']=='one_formal_round'
    assert len(records)==len(rows)==len(shapes)==1178
    assert {r['ID'] for r in records}==set(range(1,1179))
    by_id={r['ID']:r for r in records};csv_by_id={int(r['ID']):r for r in rows}
    assert len(csv_by_id)==1178
    for shape in shapes:
        idx=shape['id'];r=by_id[idx];c=csv_by_id[idx]
        for key,raw in [('M','M'),('N','N'),('K','K'),('E','G')]:
            assert int(r[key])==int(c[key])==shape[raw],(idx,key)
        assert r['phase']==c['phase']==shape['phase']
        assert c['dtype']=='w8a8_int8' and c['layout']=='NT'
        assert c['output dtype']=='int32' and c['scale']=='none'
        for backend in ['CK','BLAS']:
            assert r[backend+' status']==c[backend+' status']=='PASS'
            check=r['ck_check' if backend=='CK' else 'blas_check']
            assert check['bad_elements']==0 and check['output_guards'] and check['input_samples_unchanged']
            ms=float(r[backend+' ms']);tops=float(r[backend+' TOPS'])
            assert ms>0 and math.isfinite(ms)
            assert math.isclose(tops,shape['ops']/(ms*1e9),rel_tol=1e-12)
            assert math.isclose(ms,float(c[backend+' ms']),rel_tol=1e-12)
            assert math.isclose(tops,float(c[backend+' TOPS']),rel_tol=1e-12)
            ts=r['ck_samples_ms' if backend=='CK' else 'blas_samples_ms']
            assert len(ts)==manifest['samples'] and statistics.median(ts)==ms
        assert r['blas_info']
        assert math.isclose(r['R'],r['BLAS ms']/r['CK ms'],rel_tol=1e-12)
        assert math.isclose(r['R'],float(c['R']),rel_tol=1e-12)
        assert r['status']==c['status']==('PASS' if r['R']>=.5 else 'BELOW_TARGET')
    for path,want in manifest['binaries'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==want
    config=base/'results/full_01/run_config.json'
    fingerprint=json.loads(config.read_text())['build']['fingerprint']
    blas=(base.parent/'hipblaslt-install/lib/libhipblaslt.so.0.10').resolve()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(blas)==fingerprint['blas_sha256']
    blas_identity=dict(path=str(blas),sha256=sha(blas),asserted_against_baseline_at_startup=True,adapter_sha256=sha(base/'build/libblas_int8.so'),tuning_sha256={p:sha(Path(p)) for p in manifest['baseline_tuning_sources']})
    below=[r for r in records if r['R']<.5]
    phases={phase:dict(total=sum(r['phase']==phase for r in records),numeric_pass=sum(r['phase']==phase for r in records),ge50=sum(r['phase']==phase and r['R']>=.5 for r in records),median_R=statistics.median(r['R'] for r in records if r['phase']==phase)) for phase in ['prefill','decode']}
    audit=dict(blas_identity=blas_identity,auditor_sha256=sha(Path(__file__)),status='TARGET_MET' if not below else 'NEEDS_OPTIMIZATION',total=1178,numeric_pass=1178,ge50=1178-len(below),below50=len(below),memory_skips=0,failures=0,shape_fields_verified=True,csv_metrics_verified=True,binary_identity_verified=True,phases=phases,min_R=min(r['R'] for r in records),median_R=statistics.median(r['R'] for r in records),worst_ids=[r['ID'] for r in sorted(records,key=lambda x:x['R'])[:20]],below_target=[{k:r[k] for k in ['ID','phase','M','N','K','E','CK ms','BLAS ms','R','instance_id']} for r in below],csv_sha256=hashlib.sha256((out/'results.csv').read_bytes()).hexdigest())
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    report=['# INT8 NT Grouped GEMM 正式验收记录','',f"状态：{audit['status']}。1178条完整shape，数值通过1178条，R>=0.5为{audit['ge50']}条。无显存跳过或缺失指标。",'',
      '口径：INT8×INT8→INT32、无scale，NT，物理卡7；保持独立专家权重和完整M/N/K/E。公开默认select/query/run；BLAS使用指定安装包及已tuning的最快有效算法，同轮重新计时。','',
      f"计时：正式一轮；每样本预热{manifest['warmup']}次、计时{manifest['repeat']}次，共{manifest['samples']}个HIP event样本取中位数。",'',
      '| phase | shape数 | 数值通过 | R>=0.5 | R中位数 |','|---|---:|---:|---:|---:|']
    for phase,p in phases.items():report.append(f"| {phase} | {p['total']} | {p['numeric_pass']} | {p['ge50']} | {p['median_R']:.4f} |")
    report+=['',f"最小R={audit['min_R']:.6f}，总体R中位数={audit['median_R']:.4f}。R=BLAS ms/CK ms，TOPS=2×E×M×N×K/(ms×10^9)。",'',
      '正确性：每组全输出与独立精确可表达的FP32整数参考比较；输入范围[-8,8]，检查完整输出guard及输入抽样。完整范围[-128,127]、ragged/empty、stride/tail、非默认stream和错误参数由另存的FP64边界回归覆盖；这些验证结论不代表量化INT8输出或带scale路径已实现。','',
      '文件：results.csv、records.jsonl、manifest.json、audit.json；二进制和源码检查点在binaries/及source_checkpoint/。R未达标的shape会保留完整数值和性能记录，不从分母删除。']
    if below:
        report+=['','未达标ID：'+', '.join(str(r['ID']) for r in below)]
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
