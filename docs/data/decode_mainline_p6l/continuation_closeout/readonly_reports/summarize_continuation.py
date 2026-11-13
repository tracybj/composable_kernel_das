"""Reporting only: summarize separate frozen experiments without mixing winners."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
STAGES = [
    ('phase', '先计算与分组调度', 'phase_discovery_analysis.json', 'continuation_phase'),
    ('k64', 'K64 双槽', 'k64_discovery_analysis.json', 'continuation_k64'),
    ('n256', 'N256 双槽／三槽', 'n256_discovery_analysis.json', 'continuation_n256'),
    ('b_reuse', 'N256 wave 内 B 复用', 'b_reuse_discovery_analysis.json', 'continuation_b_reuse'),
    ('low_lds', '低 LDS N128 与自动属性规则', 'low_lds_discovery_analysis.json', 'continuation_low_lds'),
    ('wave4', 'N128 四 wave 完整 M 与 B 复用', 'wave4_discovery_analysis.json', 'continuation_wave4'),
    ('noslc', '固定结构的 A 加载缓存位修订', 'noslc_discovery_analysis.json', 'continuation_noslc'),
    ('guard', 'host 回退检查修复（7控制＋3主线）', 'guard_discovery_analysis.json', 'continuation_guard'),
    ('accum', '固定结构的双 FP32 累加缓冲', 'accum_discovery_analysis.json', 'continuation_accum'),
]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    stages, rows, packages = [], [], []
    for key, title, filename, archive in STAGES:
        path = ROOT / filename
        item = dict(stage=key, title=title, analysis=str(path), archive=archive,
                    complete=path.exists() and read(path).get('complete_phase', False))
        if path.exists():
            analysis = read(path)
            item.update(analysis_sha256=sha(path), completed_pairs=analysis['completed_pairs'])
            for group in analysis['groups']:
                for name, values in group['variants'].items():
                    row = dict(stage=key, variant=name, **values,
                               benchmark_sha256=group['binary_sha256'],
                               plan_sha256=group['plan_sha256'], epoch=group['epoch'])
                    rows.append(row)
        stages.append(item)
    for label in ['9', '10', '10b', '11', '12', '13', '14', '15', '16']:
        receipt = ROOT / f'profile_delivery_batch{label}.json'
        if receipt.exists():
            item = read(receipt)
            path = Path(item['path'])
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            packages.append(dict(batch=label, **item, receipt_sha256=sha(receipt)))
    selection_path = ROOT / 'accum_candidate_selection.json'
    if not selection_path.exists():
        selection_path = ROOT / 'noslc_candidate_selection.json'
    if not selection_path.exists():
        selection_path = ROOT / 'wave4_candidate_selection.json'
    if not selection_path.exists():
        selection_path = ROOT / 'low_lds_candidate_selection.json'
    final_active = ROOT / 'final_active.json'
    final_report = None
    if final_active.exists():
        path = Path(read(final_active)['path']) / 'report.json'
        if path.exists():
            final_report = dict(path=str(path), sha256=sha(path), report=read(path))
    accepted = final_report['report'] if final_report else None
    replacements = sum(x['original'] and x['family_adopted'] for x in accepted['rows']) if accepted else 0
    stable = accepted['stable_weak_originals'] if accepted else 0
    crossings = accepted['attributable_crossings'] if accepted else 0
    summary = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   reporting_only=True, GPU=False, stages=stages, groups=rows,
                   packages=packages,
                   selection=read(selection_path) if selection_path.exists() else None,
                   final_confirmation=final_report,
                   baseline='P6k frozen results and BLAS unchanged',
                   no_best_of_build_mix=True, original_case_count=1193,
                   original_decode_count=881, original_prefill_count=312,
                   projection_replacements=replacements,
                   not_a_fresh_full_shape_test=True,
                   accepted_stable50=stable, accepted_attributable30=crossings, performance_thresholds_pass=bool(accepted and accepted['goal_performance_pass']),
                   performance_Goal_complete=False)
    # Goal completion still requires auditing every task and deliverable.
    (OUT / 'stage_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    fields = ['stage', 'variant', 'pairs', 'improve_gt2', 'point_improve_ge5',
              'regress_gt2', 'within2', 'high_cv', 'geometric_candidate_over_base',
              'equal_case_time_sum_ratio', 'benchmark_sha256', 'plan_sha256', 'epoch']
    with (OUT / 'stage_summary.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    text = [
        '# P6l 新增 12 小时窗口：研究进展与证据索引', '',
        (f'独立确认通过整域门槛的原 shape 共 {replacements} 条，稳定弱项 {stable}/50、可归因跨桶 {crossings}/30；Goal 还需完成全部交付审计。' if replacements else '目前没有新的属性域通过采用验收，性能目标仍未完成。') + ' 原 P6k、冻结 BLAS、1193 shape 与原 Excel 保持不变；新的历史投影与新鲜实测分开保存。', '',
        '本窗口 T0 为 2026-11-13 02:10:44 UTC（北京时间 10:10:44）；新实验截止 12:10:44 UTC，硬截止 14:10:44 UTC。全部等待、编译和恢复均计时；旧会话期限保持。', '',
        '| 独立实验 | 变体 | 代表数 | 耗时 GM 候选/对照 | 改善 >2% | 回退 >2% | 高 CV |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        text.append(f'| {row["stage"]} | {row["variant"]} | {row["pairs"]} | '
                    f'{row["geometric_candidate_over_base"]:.6f} | {row["improve_gt2"]} | '
                    f'{row["regress_gt2"]} | {row["high_cv"]} |')
    text += [
        '', '每行是一套冻结二进制内的三轮配对发现数据，GM 小于 1 表示候选更快。不同构建的赢家不拼成最终成绩；14、19、23 条代表集也不能混称同一集合。点估计改善不等于七 block 的稳定改善或可归因跨桶。', '',
        '几何控制和深度控制保留在各阶段逐 shape CSV，不能将独立实验核对照冒充当前最佳 CK。原 BLAS 为 GPU0 冻结 Native GroupGEMM，本窗口正式 CK 为 GPU4 同卡两臂；早期新同卡 Native 诊断另列 R_live，不覆盖 R_frozen。', '',
        '已验证的局限：先计算调度没有形成整体收益；K64 减少循环和 SALU 后仍有资源／等待代价；N256 的部分局部收益不能通过整域回退门槛；B 复用减少寄存器与 LDS 指令后，ID805 对照的实测平均驻留基本不变；低 LDS 提高驻留后，指令／事务代价仍可能增加；四 wave 与缓存位修订均保留全部回退。这些证据否定相应充分性假设，不能据单一计数宣称完整根因。', '',
        '短 K／低 G 的两项原型、初始同卡 Native 诊断、先前 166 条七 block 否定证据，见旧会话 final_closeout；本窗口没有恢复其未完成队列。旧确认个体 28 条稳定改善／14 条归因跨桶不构成采用，不能填入本轮 50／30 计数。', '',
        '最终采用仍须：冻结属性域的全部原 shape（含已有赢家）、至少 24 条未参与设计的合成边界、固定七个配对 block、全输出精度与真实 guard、选核及回退，以及无确定 >2% 回退。原 seed 6411926 已用；下一 seed 6411927 仅在可确认的新候选实际冻结时消耗。', '',
        '以下 perf 包保留相对路径、代码与 SQTT sidecar。解压后先读各包 XCOMPUTE_README.md，再按 profile_manifest 打开目标 perf、dispatch 18 与对应 XCD/CU。Source 查主循环等待／DS read／MMAC，Wavefronts 对齐完整主循环；不同 wave 累计 Latency 不相加当墙钟。GUI 未在本机运行。', '',
    ]
    for package in packages:
        text.append(f'- 第 {package["batch"]} 批：[{Path(package["path"]).name}]({package["path"]})，'
                    f'{package["captures"]} 份有效捕获，'
                    + ('**部分交付，原计划8对仅完成2对；失败保留。** ' if package.get('partial') else '')
                    + f'SHA256 `{package["sha256"]}`。')
    text += ['', '本汇总为只读报告；实际状态、逐 shape TFLOPS、轮次、CV、实例、配置、epoch、源／二进制 SHA、原始日志与失败记录均保存在各阶段归档。没有 1193/881 全量 GPU，没有 push。', '']
    (OUT / 'README.md').write_text('\n'.join(text))
    print('READONLY_STAGE_SUMMARY', len(stages), 'stages', len(rows), 'groups',
          len(packages), 'verified packages; no GPU or adoption')


if __name__ == '__main__':
    main()
