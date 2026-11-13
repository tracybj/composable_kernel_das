"""Generate derived closeout tables and copy audited evidence; no GPU or build."""
from pathlib import Path
import collections, csv, datetime, hashlib, json, math, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/workspace/composable_kernel')
DEST = REPO / 'docs/data/decode_mainline_p6l/continuation_closeout'
STAGES = ['phase', 'k64', 'n256', 'b_reuse', 'low_lds', 'wave4', 'noslc', 'guard', 'accum']
BANDS = ['R_ge_0.99', 'R_0.8_to_0.99', 'R_0.5_to_0.8', 'R_0.3_to_0.5', 'R_0.2_to_0.3', 'R_lt_0.2']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def bucket(value):
    for index, threshold in enumerate([.99, .8, .5, .3, .2]):
        if value >= threshold:
            return BANDS[index]
    return BANDS[-1]


def main():
    closed = read(ROOT / 'research_closed.json')
    assert closed['state'] == 'research_closed_performance_goal_incomplete'
    raw = read(ROOT / 'readonly_reports/final_continuation_raw_audit.json')
    assert not raw['issues'] and not raw['pending_job_keys']
    assert len(raw['retained_failed_profiles']) == 1 and not raw['all_submissions_succeeded']
    assert raw['CK_failed_elements'] == 0
    assert read(ROOT / 'readonly_reports/binary_source_inventory_audit.json')['ok']
    assert read(ROOT / 'readonly_reports/user_preservation_audit.json')['tracked9_user_changes_preserved']
    assert not (ROOT / 'final_active.json').exists()
    assert not read(ROOT / 'accum_candidate_selection.json')['eligible_for_confirmation']
    assert not read(ROOT / 'noslc_candidate_selection.json')['eligible_for_confirmation']
    for name in ['noslc_controller_status.json', 'guard_controller_status.json', 'accum_controller_status.json']:
        assert read(ROOT / name)['state'] == 'complete'
    DEST.mkdir(parents=True, exist_ok=True)
    names = ['session.json', 'authorization.json', 'checkpoint.json', 'research_closed.json', 'gpu_stop.json',
             'environment_epoch.json', 'recovery_total.json', 'wave4_rejected_closed.json',
             'wave4_profile_abort_classification.json', 'noslc_candidate_selection.json',
             'accum_candidate_selection.json', 'profile_batch13_user_feedback.json']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'readonly_reports').iterdir())
              if p.suffix in ['.json', '.csv', '.md', '.py']]
    names += [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob('*_source_commit.json'))]
    names += [str(p.relative_to(ROOT)) for p in sorted(ROOT.glob('*_archive_commit.json'))]
    names += ['drafts/policy_fallback_guard/timing_analysis.json']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'drafts/policy_fallback_guard').glob('timing_*.log'))]
    for name in sorted(set(names)):
        source = ROOT / name
        assert source.is_file(), name
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    all_buckets, discoveries = [], []
    for stage in STAGES:
        analysis = read(ROOT / (stage + '_discovery_analysis.json'))
        assert analysis['complete_phase']
        discoveries.append(dict(stage=stage, pairs=analysis['completed_pairs'],
                                analysis_sha256=sha(ROOT / (stage + '_discovery_analysis.json'))))
        for variant in sorted({x['variant'] for x in analysis['rows']}):
            selected = [x for x in analysis['rows'] if x['variant'] == variant and not x['control']]
            if not selected:
                continue
            assert len({x['ID'] for x in selected}) == len(selected)
            for arm in ['base', 'candidate']:
                counts = collections.Counter(bucket(x['R_frozen_' + arm]) for x in selected)
                all_buckets.append(dict(stage=stage, variant=variant, arm=arm, cases=len(selected),
                                        adopted=False, **{name: counts[name] for name in BANDS}))
    with (DEST / 'fresh_buckets_by_stage_variant_NOT_ADOPTED.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, ['stage', 'variant', 'arm', 'cases', 'adopted', *BANDS], lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_buckets)
    baseline_path = REPO / 'docs/data/decode_depth8_replay_p6k/results.json'
    baseline_sha = sha(baseline_path)
    assert baseline_sha == '00c80c665735d59165947e5c9458e4e3cb0327df1a47dee2926fe725af4e9d3d'
    baseline = read(baseline_path)
    assert len(baseline) == 1193
    fields = ['ID', 'M', 'N', 'K', 'G', 'phase', 'CK_ms', 'CK_TFLOPS', 'frozen_BLAS_ms',
              'frozen_BLAS_TFLOPS', 'R_frozen', 'actual_instance', 'scheme', 'env', 'source',
              'source_SHA256', 'source_measured_at', 'quality_flags', 'adopted_replacement', 'fresh_1193_retest']
    with (DEST / 'projection_P6k_UNCHANGED_NOT_FRESH1193.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fields, lineterminator='\n')
        writer.writeheader()
        for row in baseline:
            flops = 2 * math.prod(row[k] for k in ['M', 'N', 'K', 'G'])
            value = {k: row[k] for k in ['ID', 'M', 'N', 'K', 'G', 'phase', 'CK_ms', 'frozen_BLAS_ms', 'R_frozen', 'actual_instance']}
            value.update(CK_TFLOPS=flops / row['CK_ms'] / 1e9,
                         frozen_BLAS_TFLOPS=flops / row['frozen_BLAS_ms'] / 1e9,
                         scheme=row['configured_scheme'], env=json.dumps(row['env'], ensure_ascii=False, sort_keys=True),
                         source=str(baseline_path), source_SHA256=baseline_sha,
                         source_measured_at=row['measured_at'], quality_flags=json.dumps(row['quality_flags'], ensure_ascii=False),
                         adopted_replacement=False, fresh_1193_retest=False)
            writer.writerow(value)
    projection_buckets = {}
    for phase, total in [('prefill', 312), ('decode', 881)]:
        selected = [x for x in baseline if x['phase'] == phase]
        assert len(selected) == total
        counter = collections.Counter(bucket(x['R_frozen']) for x in selected)
        projection_buckets[phase] = dict(cases=total, **{name: counter[name] for name in BANDS})
    assert [projection_buckets['prefill'][x] for x in BANDS] == [292, 20, 0, 0, 0, 0]
    assert [projection_buckets['decode'][x] for x in BANDS] == [604, 263, 14, 0, 0, 0]
    summary = read(ROOT / 'readonly_reports/stage_summary.json')
    assert len(summary['stages']) == 9 and len(summary['groups']) == 14
    assert summary['accepted_stable50'] == summary['accepted_attributable30'] == 0
    captures = sum(x['captures'] for x in raw['profile_deliveries'])
    assert captures == 80 and len(raw['profile_deliveries']) == 9
    result = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), research_closed=True,
                  performance_Goal_complete=False, Goal_status='active/incomplete',
                  accepted_stable50=0, accepted_attributable30=0, adopted_replacements=0,
                  discovery_pairs=sum(x['pairs'] for x in discoveries), stages=discoveries,
                  completed_jobs=raw['completed'], formal_full_comparisons=raw['CK_full_comparisons'],
                  full_output_elements=raw['CK_full_elements'], failed_precision_elements=0,
                  retained_failed_profile_jobs=raw['retained_failed_profiles'], unknown_audit_issues=raw['issues'],
                  pending_GPU_jobs=raw['pending_job_keys'], profile_packages=9, valid_SQTT_captures=80,
                  profile14_partial=True, GUI_verified=False, new_heldout_seed_drawn=False,
                  historical_projection_buckets=projection_buckets, projection_fresh=False,
                  no_best_of_build_mix=True, original_P6k_shapes_Excel_unchanged=True,
                  no_push_or_merge=True, deadlines_unchanged=True,
                  final_frozen_benchmark_SHA256='dd8e20c51f611150ada550ac634d68ff102237ee96fbba5e7dc4a590d1fc37ad')
    assert result['discovery_pairs'] == 348
    write(DEST / 'summary.json', result)
    text = ['# P6l 新增窗口收尾：性能目标未达成', '',
            '**没有新属性域通过采用验收，50条稳定改善／30条可归因跨桶均为0；P6k正式表、冻结BLAS、1193shape和原Excel不更新。Goal保持未完成。**', '',
            '三种已登记所有权结构及后续受控因素未形成可采用的广泛收益。最后固定批次完成后于11月13日11:57:58 UTC关闭新实验门禁，转只读审计；没有假称提前到达截止时间。授权新实验截止仍12:10:44 UTC、硬截止14:10:44 UTC，旧会话期限和记录保持。没有恢复P6g/P6i，也不跑1193或881全量GPU。', '',
            '| 实验因素 | 实际代表／变体 | 候选/当前CK耗时GM | 改善>2% | 回退>2% | 高CV |',
            '|---|---|---:|---:|---:|---:|']
    for row in summary['groups']:
        text.append(f'| {row["stage"]} | {row["variant"]}，{row["pairs"]}条 | {row["geometric_candidate_over_base"]:.6f} | {row["improve_gt2"]} | {row["regress_gt2"]} | {row["high_cv"]} |')
    text += ['', 'GM小于1表示候选更快。每行来自独立冻结二进制内的三轮配对；代表集合有重叠，不能跨构建逐shape取最快值。guard的3条是修复控制，不等于16条完整代表。全部348个三轮配对不是348个独特shape。', '',
             '最后的单累加缓存修订GM1.124657；双FP32累加修订GM1.123799，仍慢约12.38%，16代表有2条改善>2%、11条回退>2%、3条在±2%、2条高CV。两套FP32累加使M16/32/64的独立MMAC目标向量2/4/8变4/8/16，但寄存器升至67/99/163；求和顺序改变仍通过原阈值全输出，静态链数增加尚未转化为当前最佳CK之上的整体收益。原96shape/53弱项属性规则拒绝，未抽新seed、未启动不合格候选的扩围／七block确认。', '',
             '已证实的有限结论：K64减少SALU、B复用减少LDS指令／寄存器、N128降低LDS并提高驻留、取消A-load SLC、增加独立累加链，这些改动单独都不足以保证广泛收益。代码、资源、PMC和正式配对结果分别保留；具体动态等待／MMAC重叠仍需核对SQTT，不据高wait或单个长条宣布全部根因已解决。完整证据矩阵见readonly_reports/mechanism_matrix.md。', '',
             'host修复只跳过不可能命中的优先selector；322845项CPU的status/instance/reason一致，全部1766个shader不变，35资格和10配对验证通过。更严格同进程11块CPU对照表明修复本身仅减少约0.04～0.16微秒；先前约1微秒是跨环境开关配置差值，不能全归因这次修复。GPU短控制仍有高CV，不计明确GPU收益。纠正及全部原始CPU记录保留。', '',
             '原窗口已经完成H1/H2/H3、短K／低G两项原型和166条七block否定证据。本窗口续研348组三轮配对，旧个体28条稳定改善／14条归因跨桶因整域失败仍不能计入采用；原347计划中181项未完成、两个未配对精度和未测short确认维持历史状态，不能写成已完成。详见../final_closeout/closeout/README.md。', '',
             f'本窗口逐原始日志核对3187个作业：2976次全输出／{raw["CK_full_elements"]}个元素，精度失败0；没有待结束的GPU作业。另有1次profiler中应用rc134中止，原日志与失败状态保留并排除有效测量，不把它计作成功或已确认MKFD。第14批原8对只完成2对，部分包和稀疏尝试保留；第15/16批各4对完整完成。', '',
             '所有本窗口GPU均物理4/epoch0/统一锁、fresh hy-smi及可读pre/post dmesg门禁。无新增环境恢复；累计445.379秒来自旧窗口dmesg权限事件，不能称为已确认硬件MKFD。没有reset驱动、换卡或跨epoch拼接。9套正式二进制和224份编译输入副本SHA均核对；构建包含受保护用户overlay，不宣称clean HEAD本身就是冻结测量二进制。', '',
             '用户9处tracked贡献保留：7处改动块逐字不变；两处注册表数量随新增8个自有实例从61/63变69/71，用户原有+2贡献和条目逐字保留。10个untracked SHA及readfirstlane MD5保持，主Git index没有代为暂存用户改动。初始过严的数量逐字审计失败及纠正原样归档。', '',
             'fresh_buckets_by_stage_variant_NOT_ADOPTED.csv只统计各阶段实际代表，全部未采用；每阶段逐shape及TFLOPS、配置、实际实例、轮次、CV、R_frozen、二进制/epoch和日志键在对应continuation_*归档。projection_P6k_UNCHANGED_NOT_FRESH1193.csv是零替换历史投影，不能称新鲜全量：prefill312为292/20/0/0/0/0，decode881为604/263/14/0/0/0。冻结BLAS来自GPU0 Native GroupGEMM，原同卡Native诊断单列R_live，不覆盖冻结值。', '',
             '本窗口9个下载包共80份有效SQTT，逐成员SHA已核对。解压后先读XCOMPUTE_README，再开manifest指定perf、dispatch18和目标kernel/XCD/CU；第16批重点检查双累加寄存器及尾部FP32相加。perf时长不是正式成绩，不同wave累计时钟不能相加当墙钟。本机未运行Windows GUI；用户暂未查看第13批并要求继续，反馈不作为等待条件。', '']
    for package in summary['packages']:
        text.append(f'- 第{package["batch"]}批：[{Path(package["path"]).name}]({package["path"]})，{package["captures"]}份；SHA256 `{package["sha256"]}`。' + (' **部分交付。**' if package.get('partial') else ''))
    text += ['', '只读重建汇总命令：', '',
             f'`python3 -B {ROOT}/readonly_reports/summarize_continuation.py`', '',
             f'`python3 -B {ROOT}/readonly_reports/build_closeout.py`', '',
             '原阶段控制器按冻结记录和实例核对执行，保留脚本原版本及各job键，不覆盖旧原始数据。当前新GPU门禁已关闭；这些汇总命令不会解锁、编译、恢复旧队列或发起GPU。后续实验须另有明确范围和有效期限，不能重写本窗口session来续跑。', '',
             '全部提交仅本地、实验用、不合入；没有push。仍未实现广泛性能目标，后续工作应保留这些否定证据与真实数据口径。', '']
    (DEST / 'README.md').write_text('\n'.join(text))
    write(DEST / 'manifest.json', [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size, sha256=sha(p))
                                  for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'manifest.json'])
    print('CLOSEOUT_REPORT_READY', result['discovery_pairs'], 'pairs,0/50,0/30,noGPU', flush=True)


if __name__ == '__main__':
    main()
