"""Assemble final measured findings; no per-ID production policy is inferred."""
import collections,datetime,json,statistics,time
import runtime as r

read=lambda name:json.loads((r.ROOT/name).read_text())
pairs=read('per_shape_results.json');raw=[json.loads(x) for x in (r.ROOT/'jobs.jsonl').read_text().splitlines()]
valid=[x for x in raw if x.get('ok')];audit=read('live_audit.json');views=read('incremental_summary.json')
session=read('session.json');launch=read('launch_scope_results.json')
med=statistics.median
def scope(name):
    xs=[x for x in pairs if x['variant']==name]
    return dict(shapes=len(xs),median_paired_delta_pct=med(x['paired_delta_pct'] for x in xs),
        equal_case_total_delta_pct=med(100*(sum(x['after_round_ms'][rd] for x in xs)/sum(x['before_round_ms'][rd] for x in xs)-1) for rd in range(3)),
        gains_gt2pct=sum(x['paired_delta_pct'] < -2 for x in xs),regressions_gt2pct=sum(x['paired_delta_pct']>2 for x in xs),
        stable_gains=sum(x['quality']=='stable_direction' and x['paired_delta_pct'] < -2 for x in xs),
        stable_regressions=sum(x['quality']=='stable_direction' and x['paired_delta_pct']>2 for x in xs),
        quality_counts=dict(collections.Counter(x['quality'] for x in xs)),
        R_range=[min(x['R_after_frozen'] for x in xs),max(x['R_after_frozen'] for x in xs)])
resource=scope('k32_resource');assert resource['shapes']==159
all_regressions=[x for x in pairs if x['paired_delta_pct']>2]
(r.ROOT/'all_regressions.json').write_text(json.dumps(all_regressions,indent=2))
(r.ROOT/'k32_resource_summary.json').write_text(json.dumps(resource,indent=2))
now=time.time();elapsed=now-session['t0_epoch'];hours=elapsed/3600
summary=dict(at=r.stamp(),wall_seconds_at_report=elapsed,wall_hours_at_report=hours,
    t0_utc=session['t0_utc'],deadline_utc=datetime.datetime.fromtimestamp(session['deadline_epoch'],datetime.timezone.utc).isoformat(),
    tasks=dict(tail=dict(implemented=True,full_output_verified=True,performance_verified=True,recommendation='Adopt default-off tail256 attribute route',**scope('tail256')),
               cache=dict(implemented=True,full_output_verified=True,performance_verified=True,recommendation='Adopt bounded default-off NoSLC/M01=5 attribute route',**scope('cache_policy')),
               decode=dict(implemented=True,full_output_verified=True,performance_verified=True,recommendation='No broad adoption of new steady/tail refactor or159-case existingK32 route',
                           k16=scope('k16_steady'),k32=scope('k32_steady'),existing_k32_resource=resource)),
    audit=audit,latest_contrasts=len(pairs),all_regressions=len(all_regressions),
    unique_formal_shape_ids=sorted({x['id'] for x in pairs}),
    new15_full_output_checked=all(any(x.get('kind')=='accuracy' and x.get('id')==i for x in valid) for i in range(1179,1194)),
    launched_diagnostic_rows=len(launch),formal_batch_diagnostics_excluded=True,
    views=views,recovery=read('recovery_total.json'),
    failed_records=[dict(key=x['key'],rc=x.get('rc'),kernel_service_events=x.get('kernel_service_events',[])) for x in raw if not x.get('ok')],
    protocol_exclusions=read('measurement_exclusions.json'),
    limitations=['No fresh1193/1178 GPU replay','Default experiments remain off','Historical envelopes are numeric observed minima, not stability/optimality proofs',
                 'Long-term all1193 >=BLAS target remains unmet','No implementation/claim for2LDS/two-wave/GEMV/SplitK in this round'],
    local_commits_before_final=['9cdc853','d8ca5cf','e35b459','6bf6cd9'])
(r.ROOT/'final_summary.json').write_text(json.dumps(summary,indent=2))

def table(name,data):
    lines=[f'### {name}','', '| 阶段 | case | ≥0.99 | [0.8,0.99) | [0.5,0.8) | [0.3,0.5) | [0.2,0.3) | <0.2 | 本轮来源 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for phase in ['prefill','decode','all']:
        x=data[phase];lines.append('| '+phase+' | '+' | '.join(map(str,[x['total']]+x['bins']+[x['fresh']]))+' |')
    return '\n'.join(lines)
diag=['| ID / 实例 | CK 单 launch ms | CK 20次/样本归一 ms | 差值 μs | 同 epoch Native 单/20次 ms |',
      '|---|---:|---:|---:|---|']
for x in launch:
    ns=x['Native_same_epoch']['timings'];nt='/'.join(f"{v['median_ms']:.6f}" for v in ns)
    diag.append(f"| {x['id']} / {x['instance_id']} | {x['CK_single_ms']:.6f} | {x['CK_batch20_per_launch_ms']:.6f} | {x['amortized_difference_us']:.3f} | {nt} |")
text=f'''# P6f 最终报告（实验用，不合入）

本轮三个方向均完成了合理有限的实现、全输出精度和性能实验。推荐限定属性域的尾块实例和缓存策略；decode 新稳态/收尾流水及159条现有K32扩围不推荐整体采用。实验仍默认关闭，没有 push，也没有1193/1178全量GPU测试。长期“全部shape达到BLAS”仍未完成。

T0={session['t0_utc']}；报告冻结={summary['at']}，已用壁钟 {hours:.3f} 小时（阅读、编译、等待、恢复均计入）。固定14/16小时截止没有重置。完成有限实验后提前收尾，不为耗满预算启动无证据实验。

## 任务1：独立列尾块，推荐采用

16条 N1232/K1536/M1..128/G196或392 全输出、guard、三轮A/B完成。统一tail256等case总耗时下降 {abs(summary['tasks']['tail']['equal_case_total_delta_pct']):.3f}%，R={summary['tasks']['tail']['R_range'][0]:.3f}..{summary['tasks']['tail']['R_range'][1]:.3f}，全部超过冻结Native，无>2%退化。ID146由5.268581ms降至0.747846ms；冻结Native0.821168ms，同卡单launch0.819679ms。tail128也全部超过冻结BLAS，但不按ID混用宽度来追逐微小差异。

独立契约：common、非ragged/empty、gfx936 BF16NN、M≤128、N≥128且N%8=0、N%128≠0、64≤K≤4096且K%16=0、G≤1024，并满足代码中的地址/网格范围。B加载和C写回均处理列边界及M边界；原V7无列谓词epilogue未放宽。默认selector通过属性选择5016；不支持条件回退。

ID146 profile的MMOP同为12042240；VMEM总指令22,067,200→1,646,400，非MMAC VALU112,251,200→3,512,320，scratch132B/lane→0，实际grid80个persistent CTA→980个普通CTA。HBM只减少7.5%，L2命中率反而下降；证据支持spill/指令及调度工作减少，不能把全部收益归给padding或缓存命中。ID147的M padding使新MMOP翻倍，不能把等MMOP推广到所有M。新5个shader均零scratch/spill。

## 任务2：缓存与遍历，推荐限定采用

M01=1/5/8 × A SLC/NoSLC因素隔离完成；随后按预先冻结属性域扩围。最终CACHE_GRID实际入口覆盖8条活动shape+4条N4096边界；8条活动shape均改善11.67%..39.66%，R1.066..1.200；4条边界仍选5000，变化在噪声内。

采用域为common、非ragged/empty、gfx936 BF16NN、1024<M≤2048、N>4096且N%256=0、3072≤K≤4096且K%16=0、G≤1024，加地址/网格限制；5012使用NoSLC/M01=5。这里的“完整域”指1193附录内符合该域的全部8条，并非所有可能输入性能已测。

ID781同epoch baseline/SLC-M5/NoSLC-M5 profile的HBM81.29/60.57/31.18GB，combined L2命中55.83→79.09%，访存延迟746.1→344.9cycles；相同43008 CTA、512线程、64KiB LDS、VMEM/wave和MMOP。实际三轮耗时验证了缓存/遍历收益。N4096探索没有稳定收益，未纳入活动分支。

## 任务3：decode流水正确，但不整体采用

新5014(K32)、5015(K16)保留4级LDS、3组operand bank和必要同步，重写稳态和无global-load收尾。CPU生命周期模型S4..1024通过；短K、phase余数、最小stage、全M压力、重复运行和新增1179–1193的受影响/回退检查通过。所有15条新增shape均纳入全输出验证，不重新编号。

24条首批对照均保留，包括M128/256回退。K32在620/621对当前5003改善10.99%/13.79%，但相同几何old5008→new5014只有−1.41%/约0%，未超过2%及实测波动；不能归功于新收尾。K16大多持平/退化，新小M/K512也有退化。明确区分“候选未支持而回退5000”与真正新kernel变慢。

76 K16 VMEM/wave106.5→96.5、分支103.5→81（旧drain258.5）；密集K16分支183.5→82。K32密集VMEM209/211→194/196、分支111→75，但同几何耗时没有稳定改善，SGPR分配有所增加。计数改善不等于关键路径改善。保留LGKM0/barrier；LDS bank-conflict计数不是总耗时占比，因此没有无证据启动两级LDS或两wave。

额外冻结159条现有5008范围：M≤64、N%128=0、2048≤K≤4096且K%32=0、80≤ceil(M/64)ceil(N/128)G≤640，及common/架构/地址限制。全部159条已完成对各shape当前已知最好可用路线的新鲜三轮对照。
配对变化中位数 {resource['median_paired_delta_pct']:.3f}%，等case总耗时 {resource['equal_case_total_delta_pct']:.3f}%；>2%收益 {resource['gains_gt2pct']} 条，>2%退化 {resource['regressions_gt2pct']} 条；其中超过实测波动且三轮方向一致的收益/退化为 {resource['stable_gains']}/{resource['stable_regressions']}。整域不采用，不按赢家ID硬编码选核。所有负结果在all_regressions.json及逐shape表中。

小M NoSLC只测预定4条隔离控制；620/621改善5.56%/10.45%，但目前K32更快，76/358在噪声内；未扩大为全tile提示扫描。

## 单launch口径诊断

显式新增默认关闭的--launches-per-sample；默认仍为1，正式成绩严格warmup10/samples30、每样本单launch、include-args-build0。诊断20次使用每次公共API/default selector，不绕过选核。新基准二进制的18个设备shader与此前正式二进制逐字节相同。

{chr(10).join(diag)}

多launch差值反映计时边界、提交间隔及摊销的综合影响，不能独立证明某条固定开销的根因。完整三轮CV、时间、同epoch Native及来源见launch_scope_results.json。这些数据仅作诊断，未进入正式R、1193表或历史包络；冻结旧1178 GPU0/repeat20和新15 GPU4/repeat20保持只读，未重调算法。

ID76既有5008在同卡同单launch为0.021440ms，Native为0.024160ms；20次归一后分别0.015188/0.017080ms，两种一致口径下CK都更快。但拿CK单launch对冻结Native repeat20仍达不到R≥0.99，说明这类短延迟行的表面差距包含明显的测量口径因素。不能因此修改既定验收R。620/621的新K32在同单launch仍比Native慢约8%/10%，20次归一后仍有约9%/10%的差距；密集残余不能仅靠摊薄边界开销解释。新5014相对同几何5008在76的两种口径均未建立稳定收益。

## 1193来源标注增量汇总

R=冻结Native/正式单launch CK。以下各行互斥相加；所有视图312 prefill+881 decode=1193。策略叠加只更新已验证尾块16+缓存8，其余沿用带来源的历史逐shape当前路线，不能描述成统一策略全量复测。历史包络取历史当前可选基线与本轮完整三轮系列的数值最小值，包含未推荐实验及噪声内微小差异；历史行的单轮/环境不确定及BLAS大跨度等标签继续保留，无法恢复的旧二进制SHA留空，不代表稳定最优或生产ID oracle。

{table('本轮之前的历史当前表',views['baseline'])}

{table('已验证属性策略叠加历史表',views['views']['policy_overlay_incremental_1193'])}

{table('历史观测最优包络',views['views']['historical_envelope_incremental_1193'])}

## 验证、环境与交付

正式最新对照 {len(pairs)} 组；全输出精度作业 {audit['accuracy_jobs']}，比较 {audit['whole_output_comparisons']} 次、元素 {audit['whole_output_elements']:,}，所有已接受记录max_abs=0，guard通过。此计数不冒充互不重复shape数。28组实际profiler capture及全部原始失败保留现场，profile时间不替代正式计时。

所有GPU作业物理4、统一锁串行，逐命令新hy-smi满足rc0、无MKFD、卡存在且HCU0%、MemAvailable≥64GiB，并核对服务事件。恢复 {summary['recovery']['attempts']} 次，累计恢复相关观察/等待 {summary['recovery']['seconds']/60:.2f} 分钟。服务事件窗口数据排除，受影响shape整组三轮新epoch重测，完整旧记录保留。最初一次恢复额外做了未间隔的probe前检查，后续已修正为复用最后健康观察作为probe gate；未隐去该流程偏差。已知正确探针和连续健康不等于今后保证稳定。

另一失败为profiler过滤器没有捕获dispatch，属于软件问题，与环境异常分开；修复过滤器后新key重测，均有实际.perf。无reset/服务重启/换卡绕过、无终止他人进程。

最终CPU审计还发现ID688候选回退5000遗漏了该shape/该二进制的性能前全输出门禁（控制5009已有）。原两组三轮共12条计时按整组排除，保留原始rc0记录及初始checkpoint；measurement_exclusions.json明确其不再可作为最终验收证据。最终二进制先做5009/5000各3次全输出，再完整补测两组三轮，最终表仅使用补测组。最终审计对每个正式控制/候选都要求同shape、同实例、同二进制的全输出完成时间早于性能开始，而不只是“后来做过精度”。

四个正式CMake目标构建通过；selector/query/smoke、77288+6720属性检查、14316原选核等价检查、2880缓存因素检查、两份1193视图共2386 CPU可达性检查通过。v1原有公开token/结构及GemmTransKernelArg语义不变，仅追加实例ID；旧13shader未变，host诊断变更前后18shader未变。用户readfirstlane diff MD5与两份shape SHA保持不变，用户修改未暂存。全部实验默认关闭，未新增热路径隐式GPU分配/回读/同步。

复现入口、有效全部CK开关、二进制SHA、epoch、每轮median/CV、精度key及日志路径均在per_shape_results.csv/json。完整原记录和失败/恢复/门禁日志压缩归档；vendor ISA、code object、二进制和大型perf仅留现场。既有本地阶段提交9cdc853、d8ca5cf、e35b459、6bf6cd9；最终提交号见git log。

本轮未完成的是长期全部1193达到BLAS，以及未获新瓶颈证据的更深decode结构研究；不是把这些方向标成已优化成功。所有已授权三项的有限实现、修复、精度和性能验证已完成，没有待确认阶段。
'''
(r.ROOT/'report.md').write_text(text)
print('FINAL_SUMMARY_WRITTEN wall_hours',round(hours,3),'resource',resource)
