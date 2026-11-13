"""Report existing measurements and unchanged historical projection, CPU only."""
from pathlib import Path
import csv,json,math,statistics,time
import plan,runtime as r
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'closeout'

def write_csv(name,rows,fields=None):
    if fields is None:fields=list(rows[0])
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fields,lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(dict,list)) else row.get(k) for k in fields})

def tflops(x,ms):return 2*x['M']*x['N']*x['K']*x['G']/(ms*1e9)

def bucket(value):
    for i,limit in enumerate([.99,.8,.5,.3,.2]):
        if value>=limit:return i
    return 5

def buckets(rows,ratio_key,source):
    output=[]
    for phase in ['prefill','decode']:
        selected=[x for x in rows if x['phase']==phase]
        if not selected:continue
        counts=[0]*6
        for x in selected:counts[bucket(x[ratio_key])]+=1
        output.append(dict(source=source,phase=phase,total_case=len(selected),R_ge_099=counts[0],R_080_to_099=counts[1],R_050_to_080=counts[2],R_030_to_050=counts[3],R_020_to_030=counts[4],R_lt_020=counts[5]))
    return output

def main():
    audit=r.read(OUT/'audit.json');assert audit['ok'] and not audit['performance_Goal_complete']
    session=r.read(ROOT/'session.json');assert session['gpu_deadline_epoch']<time.time()<session['deadline_epoch']
    original=r.read(ROOT/'source_results.json');assert len(original)==1193
    variants=[]
    for stage in audit['exploratory_phases']:
        for group in stage['groups']:
            for name,data in group['variants'].items():
                variants.append(dict(stage=stage['phase'],variant=name,pairs=data['pairs'],GM_candidate_over_base=data['geometric_candidate_over_base'],GM_delta_pct=100*(data['geometric_candidate_over_base']-1),improve_gt2=data['improve_gt2'],point_improve_ge5=data['point_improve_ge5'],regress_gt2=data['regress_gt2'],within2=data['within2'],high_CV=data['high_cv'],epoch=group['epoch'],binary_sha256=group['binary_sha256'],plan_sha256=group['plan_sha256'],adopted=False))
    write_csv('research_variants.csv',variants);r.write(OUT/'research_variants.json',variants)
    measured=[]
    for x in r.read(ROOT/'final_attempts/attempt_1/results.json'):
        source=next(y for y in original if y['ID']==x['ID']);shape=x['shape'];blas=source['frozen_BLAS_ms']
        measured.append(dict(ID=x['ID'],**shape,phase='decode',baseline_instance=x['base_instance'],candidate_instance=x['candidate_instance'],baseline_ms=x['base_ms'],candidate_ms=x['candidate_ms'],baseline_TFLOPS=tflops(shape,x['base_ms']),candidate_TFLOPS=tflops(shape,x['candidate_ms']),frozen_BLAS_ms=blas,frozen_BLAS_TFLOPS=tflops(shape,blas),baseline_R_frozen=blas/x['base_ms'],candidate_R_frozen=blas/x['candidate_ms'],candidate_over_baseline=x['candidate_ms']/x['base_ms'],paired_GM=x['paired_geometric_candidate_over_base'],regress_over2=x['point_median_regression_over_2pct'],high_CV=x['high_cv'],local_stable_weak=x['count_for_stable50'],local_attributable_cross=x['count_for_attributable30'],whole_domain_rejected=True,adopted=False,count_for_Goal=False,base_blocks=x['base_blocks'],candidate_blocks=x['candidate_blocks'],binary_sha256=x['binary_sha256'],epoch=x['epoch'],raw_keys=x['keys']))
    assert len(measured)==166
    write_csv('measured_rejected_confirmation166.csv',measured)
    mb=buckets(measured,'baseline_R_frozen','fresh166_baseline')+buckets(measured,'candidate_R_frozen','fresh166_REJECTED_candidate')
    write_csv('measured_confirmation166_buckets.csv',mb)
    projected=[]
    for x in original:
        projected.append(dict(ID=x['ID'],**{k:x[k] for k in ['M','N','K','G','phase']},BLASLt_GroupGEMM_frozen_ms=x['frozen_BLAS_ms'],BLASLt_GroupGEMM_frozen_TFLOPS=tflops(x,x['frozen_BLAS_ms']),P6k_CK_ms=x['CK_ms'],P6k_CK_TFLOPS=tflops(x,x['CK_ms']),R_frozen=x['R_frozen'],actual_instance=x['actual_instance'],configuration=x['env'],P6l_replacement=False,fresh1193_replay=False,source='Historical P6k unchanged; no P6l candidate adopted',original_binary_sha256=x['binary_sha256'],original_epoch=x['epoch']))
    write_csv('projection_P6k_UNCHANGED_NOT_FRESH1193.csv',projected)
    pb=buckets(projected,'R_frozen','P6k_UNCHANGED_historical_projection')
    assert [(x['total_case'],x['R_ge_099'],x['R_080_to_099'],x['R_050_to_080']) for x in pb]==[(312,292,20,0),(881,604,263,14)]
    write_csv('projection_buckets_UNCHANGED.csv',pb)
    local_stable=sum(x['local_stable_weak'] for x in measured);local_cross=sum(x['local_attributable_cross'] for x in measured)
    criterion=dict(at=r.stamp(),target_stable_weak=50,target_attributable_cross=30,accepted_stable_weak=0,accepted_attributable_cross=0,goal_complete=False,adopted_families=[],rejected_confirmation_local_stable=local_stable,rejected_confirmation_local_cross=local_cross,local_flags_are_not_adoption=True,whole_domain_regressions=sum(x['regress_over2'] for x in measured),whole_domain_definite_all7_lowCV_regressions=r.read(ROOT/'attempt1_closure.json')['definite_regressions'],unmeasured_original_trial_pairs=181,unpaired_accuracy_jobs=2,phase_GPU_qualification_jobs=0,phase_GPU_discovery_pairs=0,new_heldout_seed_drawn=False,baseline_projection_unchanged=True)
    r.write(OUT/'performance_acceptance.json',criterion)
    requirements=[
        dict(requirement='中长K瓶颈诊断',status='research_done',evidence='16 currentCK/Native diagnostics;8 perf packages and per-profile counters/ISA',limitation='Counter observations do not prove a single universal bottleneck'),
        dict(requirement='三种主线所有权原型及有限因素',status='research_done_no_adoption',evidence='408 complete three-round pairs across6 phases;166 original seven-block confirmation',limitation='No whole adopted property domain passed the performance gate'),
        dict(requirement='短K/低G支线',status='research_done_negative',evidence='H1 N64/N32 and H2 shared-A/private-B short,layout/host factors; raw per-variant data retained',limitation='No short family adopted; original157-pair short confirmation not run'),
        dict(requirement='至少50条稳定弱项改善和30条可归因跨桶',status='NOT_ACHIEVED',evidence='0 accepted stable/0 accepted crossings',limitation='Partial winners from a rejected whole domain are excluded'),
        dict(requirement='最后调度候选实际GPU资格/性能/SQTT',status='NOT_RUN',evidence='Build9 and CPU/CFG audits only; no jobs use binary ee64d307…',limitation='27 qualification jobs,46 three-round pairs and profile9 remain unrun'),
        dict(requirement='新采用域全部原表+24未见/7block',status='NOT_DONE_NO_ADOPTED_DOMAIN',evidence='Only final attempt1 exists; second original seed not consumed',limitation='New time authorization is required for any further GPU/compiler work'),
        dict(requirement='GPU4/统一锁/MKFD/数据完整性',status='audited',evidence='6632 completed jobs,all GPU4,all recorded pre-health and pre/post kernel checks verified; recovery445.379s',limitation='This is an audit of executed work,not permission for new submissions'),
        dict(requirement='本地提交/用户改动/原始表保护',status='audited',evidence='Own-only experimental commits; readfirstlane MD5 and frozen input SHAs retained; no push',limitation='Latest source is compiled-only,not GPU-qualified'),
    ]
    r.write(OUT/'requirements_audit.json',requirements)
    text=['# P6l 收尾报告：性能目标未达成','',
        '**本轮没有可采用的新方案。50条稳定改善/30条可归因跨桶的完成数均为0；原P6k正式表、冻结BLAS和1193 shape不更新。**','',
        '首轮独立确认完成166条原表主线，耗时几何平均比为1.053153（约慢5.32%），105条中位回退超过2%，其中94条低CV且七个block均回退超过2%。因此整个域拒绝。原347配对中181项未完成，两个未配对精度作业保留；short确认未测，不能称为成功或失败。','',
        f'原表弱项中，个体满足七block稳定规则的有{local_stable}条、个体满足可归因跨桶的有{local_cross}条；**因整域失败，两者都不计入Goal，也未回填到正式表。**','',
        '## 研究与验证范围','',
        '| 阶段 | 完整发现配对 | 结论 |','|---|---:|---|',
        '| H1：A寄存器/B私有LDS、完整M |48| 主线及短N路径未获得整域收益 |',
        '| H2：A共享LDS/B私有LDS |39| 部分代表获益，仍有回退 |',
        '| H2：布局与预取顺序因素 |74| 有局部改善，未采用 |',
        '| H3：A/B协作LDS、两层预取 |85| K32代表有收益，不能外推整域 |',
        '| H3：深度及host解析因素 |116| 代表改善未通过后续166条确认 |',
        '| H3：A2/B3四wave、D4八wave资源因素 |46| GM分别1.018896/1.032618，均有明确回退 |','',
        '总计408个三轮配对，另有166个七block独立确认配对。它们有重叠shape、多个构建和epoch，不能当作574个独特shape或拼成逐shape历史最快表。完整变体结果见research_variants.csv。','',
        '主线研究使用三种所有权结构；K16/K32、深度、布局、资源及调度变化是实例/因素，不是额外的所有权架构。短线已做原型和三轮实验，得到负面结论；未得到采用方案。','',
        '## 最后调度原型：仅编译/静态证据','',
        '当前正式路径二进制为 `ee64d307fb6f506ff84326b3b2537351aee0c13ca1b1cce89604ebc302454d1d`。六目标构建于2026-11-12 17:57:39 UTC结束；已有1710个shader机器码保持，新增10个shader零spill/scratch，gfx936五个循环/寄存器依赖审计通过。宽wave M32/M64实际VGPR为73/112，四wave为63/86/133。','',
        '**没有用这个新二进制跑GPU query/smoke、全输出精度、三轮性能或SQTT，不能把寄存器减少当作性能收益。** 计划中的27项资格、46个配对、profile9以及新的完整属性域独立确认均未执行。新开关默认关闭；该代码是未验收实验。','',
        '静态设计为：先计算当前MMAC，再执行原VM/LG等待和CTA barrier，然后读取未来操作数、预取后续A/B。每wave仍完整累加K，收尾的最后ready(0)发生在该块MMAC之后、Store C之前。移植源中一条旧compact收尾注释仍写before computing，应以代码和实际CFG审计为准；为保持构建输入SHA，本轮收尾不修改源码。','',
        '## 数据口径','',
        'measured_rejected_confirmation166.csv包含本轮同二进制/epoch配对数据、TFLOPS、冻结BLAS比值和全部七block；候选明确标记REJECTED。其分桶仅代表这166条，不是新全量。','',
        'projection_P6k_UNCHANGED_NOT_FRESH1193.csv是原P6k未替换的历史投影，包含TFLOPS及逐行来源；不是本轮1193复测。互斥桶如下：','',
        '| 阶段 | 总数 | R≥0.99 | 0.8≤R<0.99 | 0.5≤R<0.8 | 0.3≤R<0.5 | 0.2≤R<0.3 | R<0.2 |','|---|---:|---:|---:|---:|---:|---:|---:|',
        '| prefill |312|292|20|0|0|0|0|','| decode |881|604|263|14|0|0|0|','',
        'R使用P6k冻结的Native BLASLt GroupGEMM耗时。16条同卡Native诊断的R_live另存，未覆盖冻结BLAS。以上是任务指定P6k基线，不是更早P6f截图中的610条桶。','',
        '## 数据与环境审计','',
        f"closeout/audit.json逐原始日志核对了{audit['GPU_jobs']}个作业：CK全输出{audit['CK_full_comparisons']}次/{audit['CK_full_elements']}个元素，Native全输出{audit['Native_full_comparisons']}次/{audit['Native_full_elements']}个元素，失败0。记录覆盖{audit['unique_original_IDs']}个原表ID，重复作业不等于独特shape。所有作业为物理GPU4，健康门禁、pre/post dmesg、默认selector实际实例、原始轮次和版本记录均通过核对。",'',
        '环境恢复1次、累计445.379秒，由dmesg权限不足触发，并非已确认的MKFD硬件故障。独立新进程按两次健康观察及固定5000探针恢复；epoch0/1分别保留，不重启驱动、不换卡，数据不跨epoch拼接。新原型没有GPU记录；全部已启动GPU作业已有终态，宿主进程名称快照另存process_snapshot.txt。未运行1193或881全量GPU测试。','',
        '## 截止与未完成原因','',
        'T0为11月12日02:53:23 UTC；新优化/编译/GPU提交截止为当日22:53:23 UTC，硬截止为11月13日02:53:23 UTC。等待、编译、中断和恢复全部计时。构建完成后接续GPU验证没有启动；恢复时已超过提交截止，所以没有补交实验或重置时钟。这是本轮未完成项，不能归因于等待用户确认，也不能伪称已验收。','',
        '后续只能在用户另行授权的新窗口中验证候选。旧session、失败记录和原验收门槛必须保留；当前继续按钮不会延长本轮期限。已准备的入口和计划不等于已跑通的一键验证，需在新会话重新检查并冻结。','',
        '## Profiler / XCompute 文件','',
        '共8批包，已重新核对大小、SHA、tar路径、README和manifest。本机未运行Windows XCompute，GUI表现仍需用户实际查看。 Source指令锚点、dispatch/XCD/CU和Wavefronts相对窗口见各包XCOMPUTE_README.md；perf时长不是正式采用成绩。','']
    for package in audit['perf_packages']:
        text.append(f"- 第{package['number']}批：[下载]({package['path']})；{package['bytes']}字节；SHA256 `{package['sha256']}`。")
    text+=['','所有提交仅本地、实验用、不合入；原用户tracked/untracked改动保留。性能目标保持未完成。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')
    r.write(OUT/'report_index.json',dict(at=r.stamp(),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=plan.sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='report_index.json'],accepted_stable50=0,accepted_cross30=0,Goal_complete=False))
    print('CLOSEOUT_REPORT_WRITTEN',len(variants),'variants;166 fresh REJECTED;1193 unchanged historical projection; performance0/0',flush=True)

if __name__=='__main__':main()
