# P6l 逐项验收复核：完整目标未达成

本页只做现有证据核对，不编译、不提交GPU、不改变门槛或重置期限。性能目标仍0/50、0/30。

明确缺口：没有可采用的广泛新流水；没有完成有效候选的全原shape＋至少24未见形状的七block独立确认；通用跨freeze的一键实验续跑行为没有完整执行验收。动态等待／MMAC重叠只在有证据的范围内说明，不能把资源变化当完整根因。

已补强核对：原Excel与原生成收据SHA一致；五个v1结构和全部原函数声明未变，GemmTransKernelArg所在头文件整体不变，旧实例编号保持；2088条正式计时和992条精度命令的协议逐条核对；最初16条Native诊断及四组CK/Native SQTT证据核对。

第一包Native的dispatch按shape分别为30/72/219/360，CK为18；后续候选包的18不能套用到Native。Native和CK单次调用设置一致，CPU提交路径差异仍保留。

旧收尾写“仅编译”的ee64d307候选已在新增窗口完成46组三轮及profile9；旧报告保留原截止时点，新报告和本索引说明状态演进。

| 条目 | 要求 | 当前证据状态 | 限制 |
|---|---|---|---|
| G01 | 至少50条当前弱项稳定改善≥5% | NOT_ACHIEVED | 接受数0；发现阶段局部赢家不替代七block整域采用。 |
| G02 | 至少30条满足逐block规则的可归因R≥0.99跨桶 | NOT_ACHIEVED | 接受数0；历史28/14个体诊断仍从接受总数排除。 |
| G03 | 拟采用整域无确定>2%回退 | NOT_ACHIEVED | 旧确认及全部新候选存在确定回退。 |
| S01 | 原始主线346/171、短线164/65完整清单 | PROVED | 本次新清单为只读重建，未假称它本身在首个GPU前冻结。 |
| S02 | 16代表的CK/Native同GPU4诊断 | PROVED_SCOPED | 单launch设置一致，CPU提交路径不同；不是GPU0/repeat20历史BLAS的重测。 |
| S03 | 区分CTA不足/驻留/HBM/B搬运/LDS/等待重叠 | PARTIAL | 资源和受控因素已有证据；完整动态根因和统一有效结构未证明。 |
| S04 | 独立流水而非V7别名 | PROVED_SCOPED | 可复用底层工具；当前实现均未通过性能采用。 |
| S05 | 完整M所有权、B复用、两层预取研究 | PROVED_RESEARCH | 完成研究不等于结构有效或已实现性能目标。 |
| S06 | 主线最多3结构、短线最多2结构，因素需依据 | PROVED_SCOPED | tile/调度/缓存/C累加银行是因素，不能按实例数称为新结构。 |
| S07 | 短K低G至少一个有依据原型及精度三轮结果 | PROVED_NEGATIVE_RESEARCH | 两种结构的短线均已做；short七block整域确认未测。 |
| S08 | 不以CTA内K分工冒充CTA数增加，不隐藏workspace/归约成本 | PROVED_SCOPED | 没有采用跨CTA K并行或新增隐式workspace。 |
| P01 | 首个GPU前固定清单、收益/回退/质量规则 | PROVED_SCOPED | 后续规则修订分别冻结；没有后验按赢家ID采用。 |
| P02 | 正式benchmark和默认selector，核对实际实例 | PROVED_EXECUTED_SCOPE | 2088计时和992精度均走正式入口；其余为query/smoke/profile。 |
| P03 | 逐条清理实验环境并记录完整配置 | PROVED_EXECUTED_SCOPE | 未将任意stride或未知地址宣传为已验证。 |
| P04 | 新增开关默认关闭，DISABLE优先，旧路径保留 | PROVED_TESTED_SCOPE | 不是1193全量GPU复测。 |
| P05 | v1 ABI和GemmTransKernelArg不变，6000..6003不重编号 | PROVED |  |
| P06 | M/N/K/G边界、尾部、禁用、N尾块/prefill/6003控制 | PROVED_TESTED_SCOPE | 只证明已登记packed/aligned契约。 |
| P07 | 先both+validate全输出，每个候选≥3次、真实CK/BLAS guard | PROVED_EXECUTED_SCOPE | 没有用257点抽样代替全输出。 |
| P08 | 精度不放宽、非有限值和guard检查、失败停队 | PROVED_EXECUTED_SCOPE | 本续研无精度失败；profiler中止按独立失败保留。 |
| P09 | 三轮AB/BA、warmup10/samples30/单launch/args0/redzone0 | PROVED_EXECUTED_SCOPE |  |
| P10 | 三轮median的median，保留min/max/CV/高CV/全部负结果 | PROVED_EXECUTED_SCOPE |  |
| P11 | 最终采用域同一个冻结二进制/配置/epoch的全部A/B | NOT_ACHIEVED | 阶段内配对合规；没有新的合格采用域，不能跨构建拼赢家。 |
| P12 | 最终域全部原shape、≥24未见、必要边界及独立清单/SHA | NOT_COMPLETED | 原24盲测未测；没有新seed或新完整独立确认。 |
| P13 | 固定7配对block，六次>2%且GM≤.95及CV规则 | PROVED_RULE_AND_PARTIAL_EXECUTION | 全域采用未通过；不能把三轮发现计入50/30。 |
| P14 | 可归因跨桶每block基线<.99且候选≥.99 | PROVED_RULE_NOT_ACHIEVED | 接受数0。 |
| D01 | 冻结BLAS和R_live分开，保留来源/卡号/repeat20 | PROVED |  |
| D02 | 新鲜实测分桶与1193历史投影分离、不只回填赢家 | PROVED | 1193投影不是新鲜1193复测。 |
| D03 | 逐shape CSV/JSON包含TFLOPS、方案配置、实例、轮次、CV、R和版本键 | PROVED_EXECUTED_SCOPE |  |
| D04 | 原P6k结果/summary、两份shape文件和原Excel只读不改 | PROVED | Excel与11月11日原生成收据SHA一致。 |
| D05 | 新Excel仅另命名且含筛选/冻结表头/TFLOPS | NOT_REQUESTED_IN_THIS_ROUND | 本轮无采用更新，未生成替代Excel。 |
| X01 | 至少4组CK/Native重点perf | PROVED | Native dispatch不能误按CK的18选择。 |
| X02 | 候选至少4组当前CK/候选，覆盖收益/平坦/回退 | PROVED_SCOPED | 第14批仍为2/8对部分交付，不冒充完成。 |
| X03 | SQTT真实trace、实际kernel/dispatch/代码核对 | PROVED_LOCAL_SCOPE | 没有本机GUI运行；不同wave累计时钟不作墙钟。 |
| X04 | 保存原名/依赖，manifest含几何/SHA/命令/开关/定位 | PROVED_LOCAL_SCOPE | GUI兼容性按未验证标注。 |
| X05 | 异步反馈，不逐阶段等待用户 | PROVED | 用户暂未查看；动态重叠未知点继续保留。 |
| E01 | 仅物理GPU4、显式HIP/CUDA4、core0、统一锁 | PROVED_RECORDED_SCOPE |  |
| E02 | 每次fresh健康门禁及pre/post可读dmesg | PROVED_RECORDED_SCOPE | 没有新环境恢复；不因普通失败认定MKFD。 |
| E03 | 恢复独立进程、最多3观察/≥60秒/两次健康/固定5000 probe | PROVED_OLD_RECOVERY_SCOPE | 本续研无环境恢复；不声称实测过新硬件MKFD。 |
| E04 | epoch内完整配对，不跨epoch混A/B | PROVED_EXECUTED_SCOPE | 旧epoch0/1和续研epoch0分别保存。 |
| E05 | 禁止reset/驱动重启/hymgr/换卡，不终止他人进程 | PROVED_RECORDED_SCOPE |  |
| E06 | 统一固定期限，等待编译恢复计时，无自行延期 | PROVED_TO_AUDIT_TIME | 硬截止14:10:44仍约束本轮CPU收尾；继续按钮不延长。 |
| E07 | 不跑1193/881全量GPU，不恢复P6g/P6i | PROVED_RECORDED_SCOPE | 1193 selector CPU核对不等于全量GPU。 |
| W01 | 保留readfirstlane、用户9 tracked+10 untracked | PROVED | 两个注册表size保持用户+2贡献，其余改动块逐字保留。 |
| W02 | 实际构建输入/二进制依赖/source_versions可追溯 | PROVED_EXECUTED_SCOPE | 构建含用户overlay，不等同clean HEAD的GPU成绩。 |
| W03 | 自有代码小证据本地提交、实验标识、不push合并 | PROVED_LOCAL_RECORDS |  |
| A01 | 诊断矩阵/设计决策/资源ISA/profile/全部淘汰理由 | PROVED_RESEARCH_SCOPE | 有效新流水与动态完整根因仍未实现。 |
| A02 | 旧“最后原型仅编译”的缺口得到后续GPU验证 | PROVED_SUPERSEDED_STATE | 旧README为旧截止时点，不改写历史。 |
| A03 | 一键报告且不修改原始测量 | PROVED |  |
| A04 | 通用一键断点实验入口完整复跑验收 | NOT_FULLY_VERIFIED | 本轮关闭且过提交截止；没有用新实验验证跨freeze/完整确认的统一续跑行为。 |
| A05 | 完成前逐项检查，不将研究完成当Goal完成 | PROVED_INCOMPLETE_GOAL | 性能数值、整域采用、24未见完整确认仍缺失。 |

完整来源见requirements.json/csv及各audit JSON。scope_reconstruction明确为当前只读重建，不能倒称为新实验前冻结；原冻结规则和manifest仍是预登记依据。

继续研究需要新的有效GPU时间授权；本次复核不改变12:10:44提交截止和14:10:44硬截止。Goal保持未完成。

补充源码复核：`resume_source_audit.json` 记录跨 freeze 续跑的具体风险。freeze 会把当前 manifest SHA 写入新 parent，随后资格／发现阶段只跳过 manifest 完全相同的旧记录；因此原控制器不能当作已验证的通用一键续跑入口。这里只读计算元数据，没有调用 freeze、没有更改测量逻辑，也没有影响现有结果。修复和真实续跑验收须在新的有效窗口内执行。
