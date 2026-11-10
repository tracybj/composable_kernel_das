"""Archive finite validation evidence without modifying kernels or historical tables."""
import ast,csv,datetime,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
import runtime as r

ROOT=Path(__file__).resolve().parent
DOC=r.REPO/'docs/data/decode_areg_scope_p6j'

def main():
    summary=json.loads((ROOT/'summary.json').read_text());decision=json.loads((ROOT/'decision.json').read_text())
    assert decision['complete'] and summary['discovery_complete']
    assert not summary['unresolved_failed_jobs'] and not summary['unresolved_incomplete_raw_keys']
    frozen=json.loads((ROOT/'scope_plan.json').read_text());audit=json.loads((ROOT/'binary_audit.json').read_text())
    lease=json.loads((ROOT/'binary_lease.json').read_text())
    assert lease['state']=='original_benchmark_restored'
    assert r.sha(r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark')==audit['saved_current_binary_sha256']
    assert r.sha(frozen['table'])==frozen['table_sha256']
    protected=subprocess.check_output(['git','-C',str(r.REPO),'diff','--','include/ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_uniform.hpp'])
    assert hashlib.md5(protected).hexdigest()=='4c7f49a5c7592767d53195a713db6b8a'
    assert json.loads(Path('/workspace/v7_coverage/p6i_b_vector/stage_pause.json').read_text())['reason']=='USER_PAUSED'
    with r.locked():pass
    rows=json.loads((ROOT/'results.json').read_text())
    narrow=decision['decisions']['n256_k512_g64']
    lines=['# A寄存器/B私有LDS八级原型：属性范围验证', '',
      '本轮按属性选取80条现有shape、24条此前未在1193表中出现的新shape及4条边界。没有运行1193全量，没有修改kernel、selector或默认开关，P6i B加载实验与P6g复现继续暂停。', '',
      '## 可重复收益的候选范围', '',
      '**gfx936、BF16 NN、packed dense，M=1～32的整数，N=256、K=512、G=64；使用八级A寄存器/B私有LDS原型。** 该规则依据算子属性，不依据shape ID。', '']
    if 'confirmation' in narrow:
        y=narrow['confirmation']
        lines += [f'独立确认覆盖全部32个M值，其中{y["improved_gt2"]}条耗时下降超过2%、{y["regressed_gt2"]}条回退超过2%；耗时变化中位数{y["median_change_pct"]:+.2f}%。逐条结果见bounded_candidate_results.csv。', '',
          f'最终状态：`{narrow["status"]}`。确认中{y["high_cv"]}条的候选或所选控制存在单轮样本CV>5%；预声明规则不允许将高CV结果直接当作已验收推广。不能把这一窄候选扩大成所有短K、小M，也没有新测Native GroupedGemm BLAS性能。', '']
    lines += [
      '## 结果与采用边界', '',
      '|预声明属性域|初筛数量|改善>2%|回退>2%|变化中位数|状态|',
      '|---|---:|---:|---:|---:|---|']
    for name,x in decision['decisions'].items():
        d=x['discovery'];median='—' if d['median_change_pct'] is None else f'{d["median_change_pct"]:+.2f}%'
        lines.append(f'|{name}|{d["cases"]}|{d["improved_gt2"]}|{d["regressed_gt2"]}|{median}|{x["status"]}|')
    lines+=['', '状态verified_bounded_scope表示通过本轮预设的有限验证；direction_confirmed_but_sample_variation_unresolved只确认重复计时方向，样本CV仍大于5%，不宣称达到推广门槛。所有候选保持默认关闭。', '',
      '属性表达式见scope_plan.json；组间可重叠，不将各行数量相加。初筛要求完整属性域中每条三轮都改善>2%、中位收益至少3%、至少4条新shape；合格域才进入独立第二次确认。确认要求无>2%回退、域中位收益至少3%，高CV继续标为不确定。', '',
      '## 现有80条与新shape', '',
      '|分组|数量|改善>2%|回退>2%|±2%|耗时变化中位数|逐case耗时和变化|高CV|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('existing','现有80条'),('unseen','新shape')]:
        x=summary[name]
        lines.append(f'|{label}|{x["cases"]}|{x["improved_gt2"]}|{x["regressed_gt2"]}|{x["within2"]}|{x["median_change_pct"]:+.2f}%|{x["sum_change_pct"]:+.2f}%|{x["high_cv"]}|')
    lines+=['', '负数表示候选更快。耗时和仅是每个case各调用一次的诊断权重，不代表生产调用频率。所有完整负结果均保留，不按赢家ID启用。', '',
      '现有shape的对照来自历史数值包络表中可启用配置，但本轮重新计时，不把历史耗时混入A/B。新shape比较四个预声明控制：V7开启的默认selector（保持真实回退）、小M K16、K16 drain、K32；以本轮中位数最小的控制作为更严格的参照。', '',
      '## 独立确认', '']
    for name,x in decision['decisions'].items():
        if 'confirmation' not in x:continue
        y=x['confirmation']
        lines.append(f'- {name}：{y["cases"]}条，改善>2% {y["improved_gt2"]}、回退>2% {y["regressed_gt2"]}，变化中位数{y["median_change_pct"]:+.2f}%，高CV {y["high_cv"]}条。')
        if x.get('all_integer_m_1_through_32_checked'):lines.append('  N256/K512/G64的确认覆盖M=1～32全部整数，包含初筛未出现的21个M值；不会只保留表现好的M。')
    if not any('confirmation' in x for x in decision['decisions'].values()):
        lines.append('没有完整属性域达到初筛要求，未启动无依据的第二轮或放宽门槛。')
    lines+=['', '## 精度、协议和版本', '',
      f'{summary["full_output_comparisons"]}次全输出比较，{summary["full_output_elements"]:,}个元素，failed=0/max_abs=0，CK/BLAS真实首尾guard均通过。所有计时臂均先通过同shape/实例/二进制的3次全输出。',
      '物理GPU4、统一锁串行、每作业fresh hy-smi空闲及可读dmesg/MKFD门禁；warmup10/samples30、单launch、include_args_build0、默认selector，轮换两臂或多臂顺序。高CV=候选或所选控制任一轮CV>5%，不按快慢删除样本或轮次。',
      '本轮性能比较只有CK两臂或多臂；逐组BLAS用于全输出精度，不把其heuristic时间当作冻结Native GroupedGemm性能。历史1193的CK/BLAS数据与分桶均未更新。',
      f'使用归档的官方benchmark，SHA {audit["archived_binary_sha256"]}，八级shader SHA {audit["depth8_isa"]["sha256"]}。其36个V7/V8控制shader与当前构建机器码逐字节相同；没有重写流水或引入新的实例。',
      '收益仅对应该归档八级原型。当前开发构建的6000已是后续A-LDS/B寄存器原型；本轮没有把八级原型重新接回当前源码的selector，不能用当前构建的同名开关代替本报告的归档程序。',
      f'为了沿用正式build/bin入口，控制器持锁临时安装该归档程序，每次退出均恢复原程序SHA {audit["saved_current_binary_sha256"]}。binary_lease.json与最终SHA检查确认已恢复；当前P6i未提交源码、原readfirstlane差异及暂停tools均保留。',
      '任意自定义stride等完整生产契约未验收；范围判断限定gfx936、packed dense BF16 NN、common非空非ragged，N整除128、K整除128及地址/grid边界。不能外推到其他精度/转置/硬件或所有未来shape。', '',
      '## 排除与修正', '',
      '新shape N128的首个控制精度作业正确回退4002，但测试脚本最初误期望5000，因此实例核对失败。其3次全输出和guard通过；停止队列并恢复原benchmark后，已用CPU selector计算全部新shape控制预期，重新运行完整组。原失败记录保留，不作成绩、不归因MKFD、不删负结果。该修正只改核对预期，没有改shape集合、kernel或计时命令。', '',
      '## 文件与实际命令', '',
      '- results.csv/results.json：所有完整组的shape、实例、三轮耗时、CV与原始日志键。',
      '- scope_plan.json、confirmation_extension.json、confirmation_plan.json：GPU前范围与独立确认清单。',
      '- decision.json、summary.json、binary_audit.json：采用判断、完整审计及旧/新二进制身份。',
      '- raw_logs.tar.gz：实际命令、环境、门禁、精度和计时日志、所有失败/排除记录。',
      '- archived_source/：归档八级原型的源快照；GPU二进制留现场，不写入Git。', '',
      '```bash',
      'python3 -B /workspace/v7_coverage/p6j_areg_scope/plan.py',
      'python3 -B /workspace/v7_coverage/p6j_areg_scope/verify_binary.py',
      'python3 -B -u /workspace/v7_coverage/p6j_areg_scope/run.py discovery',
      'python3 -B /workspace/v7_coverage/p6j_areg_scope/analyze.py',
      'python3 -B /workspace/v7_coverage/p6j_areg_scope/confirmation_plan.py',
      '# 只有确认清单非空时：run.py confirmation，然后重新analyze.py',
      'python3 -B /workspace/v7_coverage/p6j_areg_scope/decide.py',
      '```', '',
      'CPU期望修正和204项预期在selector_expected_correction.json/selector_expectations.json；脚本为现场审计副本。P6g原有期限及Goal状态没有改变。', '']
    DOC.mkdir(parents=True,exist_ok=False)
    text='\n'.join(lines);(ROOT/'report.md').write_text(text);(DOC/'README.md').write_text(text)
    bounded=sorted((x for x in rows if x['suite']=='confirmation' and x['N']==256 and x['K']==512 and x['G']==64),key=lambda x:x['M'])
    if bounded:
        columns=['ID','M','N','K','G','baseline_arm','base_instance','candidate_instance','before_ms','after_ms','change_pct','three_round_improved_gt2','high_cv','round_changes_pct','round_ms','round_cv']
        with (ROOT/'bounded_candidate_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');writer.writeheader()
            for row in bounded:writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],(dict,list)) else row[k] for k in columns})
        shutil.copyfile(ROOT/'bounded_candidate_results.csv',DOC/'bounded_candidate_results.csv')
    for pattern in ['*.py','*.cpp']:
        for p in ROOT.glob(pattern):shutil.copyfile(p,DOC/p.name)
    names=['session.json','scope_plan.json','binary_audit.json','binary_lease.json','confirmation_extension.json',
       'confirmation_plan.json','selector_expectations.json','selector_expected_correction.json','known_excluded_jobs.json',
       'results.json','results.csv','summary.json','decision.json',
       'discovery_summary.json','discovery_results.json','discovery_results.csv']
    for name in names:shutil.copyfile(ROOT/name,DOC/name)
    shutil.copytree(ROOT/'archived_source',DOC/'archived_source')
    with tarfile.open(DOC/'raw_logs.tar.gz','w:gz') as archive:
        for name in ['jobs.jsonl','started.jsonl','gate.jsonl','kernel_checks.jsonl','shape_results.jsonl','kernel_preflight.log',
                     'recovery.jsonl','waits.jsonl','environment_epoch.json','recovery_total.json']:
            p=ROOT/name
            if p.exists():archive.add(p,arcname=name)
        jobs=[json.loads(x) for x in (ROOT/'jobs.jsonl').read_text().splitlines()]
        for job in jobs:
            for name in ['command.json','run.log']:
                p=Path(job['path'])/name
                assert p.exists()
                archive.add(p,arcname=job['key']+'/'+name)
        for p in (ROOT/'scripts').iterdir():
            if p.is_file():archive.add(p,arcname='scripts/'+p.name)
    for p in DOC.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    status=dict(at=r.stamp(),complete=True,production_enabled=False,official_benchmark_restored=True,
      official_binary_sha256=r.sha(r.REPO/'build/bin/tile_example_grouped_gemm_device_args_benchmark'),
      gpu_jobs_running=0,global_lock_free=True,p6i_state='paused',p6g_state='paused',
      protected_user_diff_md5=hashlib.md5(protected).hexdigest(),shape_sha256=r.sha('/workspace/shapes_1178.csv'))
    (ROOT/'final_status.json').write_text(json.dumps(status,indent=2));shutil.copyfile(ROOT/'final_status.json',DOC/'final_status.json')
    (ROOT/'live_status.json').write_text(json.dumps(dict(at=r.stamp(),state='finite_validation_complete',physical_gpu=4,gpu_jobs_running=0),indent=2))
    manifest={str(p.relative_to(DOC)):r.sha(p) for p in sorted(DOC.rglob('*')) if p.is_file()}
    (DOC/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SCOPE_ARCHIVE_PASS files='+str(len(manifest)),flush=True)

if __name__=='__main__':main()
