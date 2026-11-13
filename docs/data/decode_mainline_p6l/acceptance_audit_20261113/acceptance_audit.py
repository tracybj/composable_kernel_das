"""Requirement-by-requirement evidence index; never runs HIP, builds, or timings."""
from pathlib import Path
import collections, csv, datetime, hashlib, json, os, re, subprocess, tarfile, time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'run_20261112T025323Z'
REPO = Path('/workspace/composable_kernel')
OUT = REPO / 'docs/data/decode_mainline_p6l/acceptance_audit_20261113'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git(*args, env=None):
    return subprocess.check_output(['git', *args], cwd=REPO, env=env)


def options(command):
    return {x[2:].split('=', 1)[0]: x.split('=', 1)[1] for x in command if x.startswith('--') and '=' in x}


def main():
    session = read(ROOT / 'session.json')
    assert time.time() < session['deadline_epoch']
    assert read(ROOT / 'research_closed.json')['new_GPU_submissions_allowed'] is False
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(x) for x in (ROOT / 'jobs.jsonl').read_text().splitlines()]
    assert len(jobs) == 3187
    counters = collections.Counter(x['kind'] for x in jobs)
    assert dict(counters) == dict(accuracy=992, perf=2088, regression=18, profile=89)
    official = str(REPO / 'build/bin/tile_example_grouped_gemm_device_args_benchmark')
    for job in jobs:
        if job['kind'] not in ['accuracy', 'perf']:
            continue
        assert job['ok'] and job['rc'] == 0 and job['command'][0] == official
        flags = options(job['command'])
        assert flags['dtype'] == 'bf16' and flags['layout'] == 'NN'
        assert flags['include-args-build'] == '0' and flags['validate'] == '1'
        assert 'force' not in ' '.join(job['command']).lower()
        if job['kind'] == 'perf':
            for key, value in dict(warmup='10', samples='30', **{'launches-per-sample': '1', 'backend': 'ck', 'time': '1', 'output-redzone': '0'}).items():
                assert flags[key] == value
        else:
            assert flags['backend'] == 'both' and flags['time'] == '0'
            assert flags['output-redzone'] == '1' and int(flags['correctness-repeats']) >= 3
    write(OUT / 'executed_protocol_audit.json', dict(
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(), ok=True, CPU_only=True, GPU=False,
        jobs_journal_sha256=sha(ROOT / 'jobs.jsonl'), kinds=dict(counters),
        all2088_formal_timings_warmup10_samples30_single_launch_args0_redzone0=True,
        all992_accuracy_jobs_backend_both_validate1_at_least3_repeats_real_redzone=True,
        actual_default_selector_checked_in_raw_audit=str(ROOT / 'readonly_reports/final_continuation_raw_audit.json')))
    baseline_file = REPO / 'docs/data/decode_depth8_replay_p6k/results.json'
    originals = read(baseline_file)
    main_scope = [x for x in originals if x['phase'] == 'decode' and x['M'] <= 64 and
                  1024 <= x['N'] <= 4096 and x['N'] % 128 == 0 and x['K'] >= 2048]
    short_scope = [x for x in originals if x['phase'] == 'decode' and x['M'] <= 256 and
                   x['K'] < 2048 and x['G'] <= 32 and x['N'] <= 4096 and x['N'] % 128 == 0]
    latest_scope = [x for x in originals if 17 <= x['M'] <= 64 and 1024 <= x['N'] <= 4096 and
                    x['N'] % 128 == 0 and 2048 <= x['K'] <= 8192 and x['K'] % 128 == 0 and 1 <= x['G'] <= 1024]
    assert (len(main_scope), sum(x['R_frozen'] < .99 for x in main_scope)) == (346, 171)
    assert (len(short_scope), sum(x['R_frozen'] < .99 for x in short_scope)) == (164, 65)
    assert (len(latest_scope), sum(x['R_frozen'] < .99 for x in latest_scope)) == (96, 53)
    assert not {x['ID'] for x in main_scope} & {x['ID'] for x in short_scope}
    write(OUT / 'scope_reconstruction.json', dict(
        at=datetime.datetime.now(datetime.timezone.utc).isoformat(), CPU_only=True, GPU=False,
        note='Reconstructed now from frozen inputs and prior predicates; this file itself was not frozen before GPU. Original research_plan and source-versioned selector contain the preregistration.',
        P6k_source=str(baseline_file), P6k_sha256=sha(baseline_file),
        initial_main=[dict(ID=x['ID'], M=x['M'], N=x['N'], K=x['K'], G=x['G'], weak=x['R_frozen'] < .99) for x in main_scope],
        initial_short=[dict(ID=x['ID'], M=x['M'], N=x['N'], K=x['K'], G=x['G'], weak=x['R_frozen'] < .99) for x in short_scope],
        latest_rejected_scope=[dict(ID=x['ID'], M=x['M'], N=x['N'], K=x['K'], G=x['G'], weak=x['R_frozen'] < .99) for x in latest_scope]))
    frozen = {
        baseline_file: '00c80c665735d59165947e5c9458e4e3cb0327df1a47dee2926fe725af4e9d3d',
        REPO / 'docs/data/decode_depth8_replay_p6k/summary.json': 'c733fb5af4471857a22ae4a1b68bfc2215745df2906b7653fbb2ce290fd8bba0',
        Path('/workspace/shapes_1178.csv'): '5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863',
        Path('/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv'): '5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863',
    }
    for path, expected in frozen.items():
        assert sha(path) == expected
    workbook_meta = Path('/workspace/v7_coverage/exports/BF16_NN_GroupGEMM_1193_对比.json')
    workbook = read(workbook_meta)
    assert sha(workbook['output']) == workbook['sha256']
    assert Path(workbook['output']).stat().st_size == workbook['bytes']
    write(OUT / 'frozen_input_and_excel_audit.json', dict(ok=True, CPU_only=True, GPU=False,
        frozen_files={str(k): v for k, v in frozen.items()}, workbook_path=workbook['output'],
        workbook_sha256=workbook['sha256'], original_workbook_receipt=str(workbook_meta),
        original_workbook_receipt_sha256=sha(workbook_meta), workbook_original_date=workbook['at']))
    env = os.environ.copy()
    env['GIT_INDEX_FILE'] = str(ROOT / 'readonly_reports/initial_user_reconstruction.index')
    name = 'include/ck_tile/ops/gemm/grouped_gemm_device_args.h'
    before = git('show', ':' + name, env=env).decode()
    after = (REPO / name).read_text()
    structs = lambda text: {m[0]: m[1] for m in re.findall(r'typedef\s+struct\s+(\w+)\s*(\{.*?\}\s*\w+\s*;)', text, re.S)}
    prototypes = lambda text: re.findall(r'\bint\s+ck_tile_hcu_\w+\s*\(.*?\)\s*;', text, re.S)
    assert len(structs(before)) == 5 and structs(before) == structs(after)
    assert prototypes(before) and prototypes(before) == prototypes(after)
    enum = lambda text: {k: int(v) for k, v in re.findall(r'(CK_TILE_HCU_GROUPED_GEMM_INSTANCE_\w+)\s*=\s*(\d+)', text)}
    enum_before, enum_after = enum(before), enum(after)
    assert all(enum_after[k] == v for k, v in enum_before.items())
    assert len(enum_after) - len(enum_before) == 8
    kernel = 'include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp'
    assert git('show', session['head'] + ':' + kernel) == (REPO / kernel).read_bytes()
    write(OUT / 'public_abi_and_instance_identity_audit.json', dict(ok=True, CPU_only=True, GPU=False,
        all5_v1_struct_bodies_byte_identical=True, all_v1_prototypes_byte_identical=True,
        all_initial_instance_names_and_numbers_preserved=True, appended_own_instance_entries=8,
        grouped_gemm_kernel_entire_header_unchanged_from_initial_HEAD=True,
        GemmTransKernelArgImpl_unchanged=True, public_current_sha256=sha(REPO / name),
        grouped_kernel_sha256=sha(REPO / kernel)))
    diag = read(OLD / 'diagnostic_analysis.json')
    oldjobs = {x['key']: x for x in map(json.loads, (OLD / 'jobs.jsonl').read_text().splitlines())}
    assert diag['complete'] and len(diag['rows']) == 16
    for row in diag['rows']:
        records = [oldjobs[k] for k in row['keys']]
        assert len(records) == 8 and all(x['ok'] and x['physical_gpu'] == 4 and x['epoch'] == row['epoch'] for x in records)
        for rec in records:
            if rec['kind'] == 'native_perf':
                native = rec['native']
                assert native['gpu'] == 4 and native['accuracy_pass']
                assert native['repeat'] == 1 and native['warmup'] == 10
                if rec['mode'] == 'accuracy':
                    assert len(native['comparisons']) >= 3
                    assert all(x['failed'] == 0 and x['finite'] and x['guards_ok'] for x in native['comparisons'])
                else:
                    assert native['samples'] == 30
    first = read(OLD / 'profile_delivery_batch1.json')
    assert Path(first['path']).stat().st_size == first['bytes'] and sha(first['path']) == first['sha256']
    with tarfile.open(first['path']) as archive:
        member = next(x for x in archive.getmembers() if x.name.endswith('/profile_manifest.json'))
        captures = json.loads(archive.extractfile(member).read())
    assert len(captures) == 8
    for ident in [76, 621, 805, 1072]:
        pair = [x for x in captures if x['ID'] == ident]
        assert len(pair) == 2 and {x['arm'] for x in pair} == {'current_best', 'frozen_native'}
        assert all(x['physical_gpu'] == 4 and x['sqtt'] and x['raw_token_bytes'] >= 4096 for x in pair)
    write(OUT / 'native_diagnostic_and_first_delivery_audit.json', dict(ok=True, CPU_only=True, GPU=False,
        matched_sampling_settings=True, shapes=16, native_repeat=1, warmup=10, samples=30,
        different_CPU_enqueue_paths_disclosed=diag['note'],
        R_live_not_frozen_BLAS=True, first_CK_Native_pairs=4, first_package_sha256=first['sha256'],
        Native_dispatch_by_ID={x['ID']: x['selected_dispatch'] for x in captures if x['arm'] == 'frozen_native'},
        CK_dispatch=18, GUI_verified=False))
    old_final = read(OLD / 'final_attempts/attempt_1/report.json')
    assert not old_final['complete'] and not old_final['goal_performance_pass']
    assert old_final['families']['main']['measured'] == 166 and old_final['families']['main']['expected'] == 190
    assert old_final['families']['short']['measured'] == 0
    variants = read(REPO / 'docs/data/decode_mainline_p6l/final_closeout/closeout/research_variants.json')
    short = [x for x in variants if x['variant'].startswith('short')]
    assert any(x['stage'] == 'H1' and x['pairs'] == 9 for x in short)
    assert any(x['stage'] == 'H2' and x['pairs'] == 9 for x in short)
    write(OUT / 'historical_confirmation_and_short_branch_audit.json', dict(
        ok=True, CPU_only=True, GPU=False, final_attempt1_complete=False,
        final_attempt1_main_measured=166, final_attempt1_main_expected190_including24fresh=190,
        final_attempt1_short_measured=0, whole_domain_adopted=False,
        stable_weak_originals_accepted=0, attributable_crossings_accepted=0, short_variants=short))
    requirement_rows = []
    def item(key, requirement, state, evidence, limitation=''):
        requirement_rows.append(dict(id=key, requirement=requirement, state=state, evidence=evidence, limitation=limitation))
    item('G01', '至少50条当前弱项稳定改善≥5%', 'NOT_ACHIEVED', 'continuation_closeout/summary.json；旧final_attempts/attempt_1/report.json', '接受数0；发现阶段局部赢家不替代七block整域采用。')
    item('G02', '至少30条满足逐block规则的可归因R≥0.99跨桶', 'NOT_ACHIEVED', '同G01', '接受数0；历史28/14个体诊断仍从接受总数排除。')
    item('G03', '拟采用整域无确定>2%回退', 'NOT_ACHIEVED', 'accum_candidate_selection.json、旧report.json', '旧确认及全部新候选存在确定回退。')
    item('S01', '原始主线346/171、短线164/65完整清单', 'PROVED', 'scope_reconstruction.json；原research_plan.json', '本次新清单为只读重建，未假称它本身在首个GPU前冻结。')
    item('S02', '16代表的CK/Native同GPU4诊断', 'PROVED_SCOPED', 'native_diagnostic_and_first_delivery_audit.json', '单launch设置一致，CPU提交路径不同；不是GPU0/repeat20历史BLAS的重测。')
    item('S03', '区分CTA不足/驻留/HBM/B搬运/LDS/等待重叠', 'PARTIAL', 'continuation_closeout/readonly_reports/mechanism_matrix.md；各profile_analysis', '资源和受控因素已有证据；完整动态根因和统一有效结构未证明。')
    item('S04', '独立流水而非V7别名', 'PROVED_SCOPED', 'gemm_pipeline_decode_nn_mainline.hpp、shared.hpp、cooperative.hpp及实际独立kernel ISA', '可复用底层工具；当前实现均未通过性能采用。')
    item('S05', '完整M所有权、B复用、两层预取研究', 'PROVED_RESEARCH', 'H1/H2/H3和9续研阶段源码、模型、实际CFG/资源/三轮数据', '完成研究不等于结构有效或已实现性能目标。')
    item('S06', '主线最多3结构、短线最多2结构，因素需依据', 'PROVED_SCOPED', 'research_variants.json；mechanism_matrix.md', 'tile/调度/缓存/C累加银行是因素，不能按实例数称为新结构。')
    item('S07', '短K低G至少一个有依据原型及精度三轮结果', 'PROVED_NEGATIVE_RESEARCH', 'historical_confirmation_and_short_branch_audit.json', '两种结构的短线均已做；short七block整域确认未测。')
    item('S08', '不以CTA内K分工冒充CTA数增加，不隐藏workspace/归约成本', 'PROVED_SCOPED', '独立源码与短线否定证据；public_abi_and_instance_identity_audit.json', '没有采用跨CTA K并行或新增隐式workspace。')
    item('P01', '首个GPU前固定清单、收益/回退/质量规则', 'PROVED_SCOPED', '原research_plan.json、immutable manifests及source_versions', '后续规则修订分别冻结；没有后验按赢家ID采用。')
    item('P02', '正式benchmark和默认selector，核对实际实例', 'PROVED_EXECUTED_SCOPE', 'executed_protocol_audit.json；3187原始记录审计', '2088计时和992精度均走正式入口；其余为query/smoke/profile。')
    item('P03', '逐条清理实验环境并记录完整配置', 'PROVED_EXECUTED_SCOPE', 'runtime/plan source_versions；qualification audits；job environment', '未将任意stride或未知地址宣传为已验证。')
    item('P04', '新增开关默认关闭，DISABLE优先，旧路径保留', 'PROVED_TESTED_SCOPE', '322845 CPU策略对照、1193 CPU baseline、资格禁用/回退', '不是1193全量GPU复测。')
    item('P05', 'v1 ABI和GemmTransKernelArg不变，6000..6003不重编号', 'PROVED', 'public_abi_and_instance_identity_audit.json')
    item('P06', 'M/N/K/G边界、尾部、禁用、N尾块/prefill/6003控制', 'PROVED_TESTED_SCOPE', '各qualification_plan/audit和逐shape控制行', '只证明已登记packed/aligned契约。')
    item('P07', '先both+validate全输出，每个候选≥3次、真实CK/BLAS guard', 'PROVED_EXECUTED_SCOPE', 'executed_protocol_audit.json及raw audit：992 accuracy/2976比较', '没有用257点抽样代替全输出。')
    item('P08', '精度不放宽、非有限值和guard检查、失败停队', 'PROVED_EXECUTED_SCOPE', '正式验证代码及parse_output source_versions、资格日志', '本续研无精度失败；profiler中止按独立失败保留。')
    item('P09', '三轮AB/BA、warmup10/samples30/单launch/args0/redzone0', 'PROVED_EXECUTED_SCOPE', 'executed_protocol_audit.json及各analysis逐记录序列检查')
    item('P10', '三轮median的median，保留min/max/CV/高CV/全部负结果', 'PROVED_EXECUTED_SCOPE', '各阶段analysis raw-pair verifier和paired_results_with_tflops.csv')
    item('P11', '最终采用域同一个冻结二进制/配置/epoch的全部A/B', 'NOT_ACHIEVED', '各scope gate及final_attempt1/report.json', '阶段内配对合规；没有新的合格采用域，不能跨构建拼赢家。')
    item('P12', '最终域全部原shape、≥24未见、必要边界及独立清单/SHA', 'NOT_COMPLETED', '旧plan main190，实测166；新96/53规则未通过发现', '原24盲测未测；没有新seed或新完整独立确认。')
    item('P13', '固定7配对block，六次>2%且GM≤.95及CV规则', 'PROVED_RULE_AND_PARTIAL_EXECUTION', '旧166原shape完整7block；heldout_protocol.json', '全域采用未通过；不能把三轮发现计入50/30。')
    item('P14', '可归因跨桶每block基线<.99且候选≥.99', 'PROVED_RULE_NOT_ACHIEVED', 'acceptance.py source_versions、旧report', '接受数0。')
    item('D01', '冻结BLAS和R_live分开，保留来源/卡号/repeat20', 'PROVED', 'native_diagnostic_and_first_delivery_audit.json；P6k逐行来源')
    item('D02', '新鲜实测分桶与1193历史投影分离、不只回填赢家', 'PROVED', 'fresh_buckets_by_stage_variant_NOT_ADOPTED.csv；零替换projection', '1193投影不是新鲜1193复测。')
    item('D03', '逐shape CSV/JSON包含TFLOPS、方案配置、实例、轮次、CV、R和版本键', 'PROVED_EXECUTED_SCOPE', '各continuation_* paired_results_with_tflops.csv、jobs.json.gz、raw tar')
    item('D04', '原P6k结果/summary、两份shape文件和原Excel只读不改', 'PROVED', 'frozen_input_and_excel_audit.json', 'Excel与11月11日原生成收据SHA一致。')
    item('D05', '新Excel仅另命名且含筛选/冻结表头/TFLOPS', 'NOT_REQUESTED_IN_THIS_ROUND', '原xlsx及其validation.json保留', '本轮无采用更新，未生成替代Excel。')
    item('X01', '至少4组CK/Native重点perf', 'PROVED', '首包8 capture/4 pair，Native dispatch30/72/219/360', 'Native dispatch不能误按CK的18选择。')
    item('X02', '候选至少4组当前CK/候选，覆盖收益/平坦/回退', 'PROVED_SCOPED', '第15/16批各4对，预登记rule及manifest', '第14批仍为2/8对部分交付，不冒充完成。')
    item('X03', 'SQTT真实trace、实际kernel/dispatch/代码核对', 'PROVED_LOCAL_SCOPE', 'trace inventories、captured ISA、80有效续研capture', '没有本机GUI运行；不同wave累计时钟不作墙钟。')
    item('X04', '保存原名/依赖，manifest含几何/SHA/命令/开关/定位', 'PROVED_LOCAL_SCOPE', '各package XCOMPUTE_README及profile_manifest', 'GUI兼容性按未验证标注。')
    item('X05', '异步反馈，不逐阶段等待用户', 'PROVED', 'profile_batch13_user_feedback.json及后续作业时间', '用户暂未查看；动态重叠未知点继续保留。')
    item('E01', '仅物理GPU4、显式HIP/CUDA4、core0、统一锁', 'PROVED_RECORDED_SCOPE', 'runtime版本、各job env、3187原始审计')
    item('E02', '每次fresh健康门禁及pre/post可读dmesg', 'PROVED_RECORDED_SCOPE', 'gates/kernel_checks journals与raw audits', '没有新环境恢复；不因普通失败认定MKFD。')
    item('E03', '恢复独立进程、最多3观察/≥60秒/两次健康/固定5000 probe', 'PROVED_OLD_RECOVERY_SCOPE', '旧recovery日志与audit；445.379秒单次恢复', '本续研无环境恢复；不声称实测过新硬件MKFD。')
    item('E04', 'epoch内完整配对，不跨epoch混A/B', 'PROVED_EXECUTED_SCOPE', '各pair二进制/epoch元组原始检查', '旧epoch0/1和续研epoch0分别保存。')
    item('E05', '禁止reset/驱动重启/hymgr/换卡，不终止他人进程', 'PROVED_RECORDED_SCOPE', '已执行GPU命令和恢复脚本；仅自身已结束PID只读快照')
    item('E06', '统一固定期限，等待编译恢复计时，无自行延期', 'PROVED_TO_AUDIT_TIME', 'session.json及所有作业/构建时间；12:10:44新实验截止', '硬截止14:10:44仍约束本轮CPU收尾；继续按钮不延长。')
    item('E07', '不跑1193/881全量GPU，不恢复P6g/P6i', 'PROVED_RECORDED_SCOPE', '冻结有限清单、jobs IDs、session pause fields', '1193 selector CPU核对不等于全量GPU。')
    item('W01', '保留readfirstlane、用户9 tracked+10 untracked', 'PROVED', 'user_preservation_audit.json', '两个注册表size保持用户+2贡献，其余改动块逐字保留。')
    item('W02', '实际构建输入/二进制依赖/source_versions可追溯', 'PROVED_EXECUTED_SCOPE', '9正式二进制、224输入副本SHA、raw source_versions', '构建含用户overlay，不等同clean HEAD的GPU成绩。')
    item('W03', '自有代码小证据本地提交、实验标识、不push合并', 'PROVED_LOCAL_RECORDS', 'own-only private-index提交收据；Git blob校验')
    item('A01', '诊断矩阵/设计决策/资源ISA/profile/全部淘汰理由', 'PROVED_RESEARCH_SCOPE', 'mechanism_matrix、两次窗口报告、逐阶段archive', '有效新流水与动态完整根因仍未实现。')
    item('A02', '旧“最后原型仅编译”的缺口得到后续GPU验证', 'PROVED_SUPERSEDED_STATE', 'phase_discovery_analysis：ee64d307…，405作业/46三轮', '旧README为旧截止时点，不改写历史。')
    item('A03', '一键报告且不修改原始测量', 'PROVED', 'summarize_continuation.py/build_closeout.py及Git数值验证收据')
    item('A04', '通用一键断点实验入口完整复跑验收', 'NOT_FULLY_VERIFIED', '阶段控制器与runtime job键可追溯', '本轮关闭且过提交截止；没有用新实验验证跨freeze/完整确认的统一续跑行为。')
    item('A05', '完成前逐项检查，不将研究完成当Goal完成', 'PROVED_INCOMPLETE_GOAL', '本审计及0/50、0/30，Goal active', '性能数值、整域采用、24未见完整确认仍缺失。')
    write(OUT / 'requirements.json', requirement_rows)
    with (OUT / 'requirements.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, ['id', 'requirement', 'state', 'evidence', 'limitation'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(requirement_rows)
    totals = collections.Counter(x['state'] for x in requirement_rows)
    write(OUT / 'summary.json', dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        CPU_only=True, GPU=False, required_items=len(requirement_rows), states=dict(totals),
        full_objective_achieved=False, accepted_stable50=0, accepted_attributable30=0,
        new_GPU_submission_deadline_passed=time.time() >= session['gpu_deadline_epoch'],
        no_deadline_extension=True, Goal_complete=False))
    text = ['# P6l 逐项验收复核：完整目标未达成', '',
        '本页只做现有证据核对，不编译、不提交GPU、不改变门槛或重置期限。性能目标仍0/50、0/30。', '',
        '明确缺口：没有可采用的广泛新流水；没有完成有效候选的全原shape＋至少24未见形状的七block独立确认；通用跨freeze的一键实验续跑行为没有完整执行验收。动态等待／MMAC重叠只在有证据的范围内说明，不能把资源变化当完整根因。', '',
        '已补强核对：原Excel与原生成收据SHA一致；五个v1结构和全部原函数声明未变，GemmTransKernelArg所在头文件整体不变，旧实例编号保持；2088条正式计时和992条精度命令的协议逐条核对；最初16条Native诊断及四组CK/Native SQTT证据核对。', '',
        '第一包Native的dispatch按shape分别为30/72/219/360，CK为18；后续候选包的18不能套用到Native。Native和CK单次调用设置一致，CPU提交路径差异仍保留。', '',
        '旧收尾写“仅编译”的ee64d307候选已在新增窗口完成46组三轮及profile9；旧报告保留原截止时点，新报告和本索引说明状态演进。', '',
        '| 条目 | 要求 | 当前证据状态 | 限制 |', '|---|---|---|---|']
    for row in requirement_rows:
        text.append(f'| {row["id"]} | {row["requirement"]} | {row["state"]} | {row["limitation"]} |')
    text += ['', '完整来源见requirements.json/csv及各audit JSON。scope_reconstruction明确为当前只读重建，不能倒称为新实验前冻结；原冻结规则和manifest仍是预登记依据。', '',
             '继续研究需要新的有效GPU时间授权；本次复核不改变12:10:44提交截止和14:10:44硬截止。Goal保持未完成。', '']
    (OUT / 'README.md').write_text('\n'.join(text))
    shutil_source = Path(__file__)
    (OUT / shutil_source.name).write_bytes(shutil_source.read_bytes())
    write(OUT / 'manifest.json', [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size, sha256=sha(p))
                                for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'manifest.json'])
    print('REQUIREMENTS_AUDIT_WRITTEN', len(requirement_rows), 'items; complete=False; noGPU', flush=True)


if __name__ == '__main__':
    main()
