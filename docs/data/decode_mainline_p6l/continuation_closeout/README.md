# P6l 新增窗口收尾：性能目标未达成

**没有新属性域通过采用验收，50条稳定改善／30条可归因跨桶均为0；P6k正式表、冻结BLAS、1193shape和原Excel不更新。Goal保持未完成。**

三种已登记所有权结构及后续受控因素未形成可采用的广泛收益。最后固定批次完成后于11月13日11:57:58 UTC关闭新实验门禁，转只读审计；没有假称提前到达截止时间。授权新实验截止仍12:10:44 UTC、硬截止14:10:44 UTC，旧会话期限和记录保持。没有恢复P6g/P6i，也不跑1193或881全量GPU。

| 实验因素 | 实际代表／变体 | 候选/当前CK耗时GM | 改善>2% | 回退>2% | 高CV |
|---|---|---:|---:|---:|---:|
| phase | compact，19条 | 1.015421 | 8 | 4 | 2 |
| phase | wide，14条 | 1.031505 | 6 | 8 | 0 |
| k64 | compact，19条 | 1.081196 | 5 | 11 | 12 |
| k64 | wide，14条 | 1.136593 | 1 | 10 | 3 |
| n256 | d2，19条 | 1.042323 | 7 | 11 | 0 |
| n256 | d3，19条 | 1.034429 | 4 | 8 | 1 |
| b_reuse | d2，19条 | 1.015139 | 9 | 7 | 1 |
| b_reuse | d3，19条 | 1.034028 | 6 | 10 | 3 |
| low_lds | auto，23条 | 1.089295 | 8 | 10 | 2 |
| low_lds | pure，23条 | 1.173542 | 1 | 18 | 0 |
| wave4 | scope_wave4，16条 | 1.183817 | 1 | 13 | 1 |
| noslc | noslc，16条 | 1.124657 | 2 | 12 | 2 |
| guard | noslc，3条 | 1.171838 | 1 | 2 | 1 |
| accum | accum，16条 | 1.123799 | 2 | 11 | 2 |

GM小于1表示候选更快。每行来自独立冻结二进制内的三轮配对；代表集合有重叠，不能跨构建逐shape取最快值。guard的3条是修复控制，不等于16条完整代表。全部348个三轮配对不是348个独特shape。

最后的单累加缓存修订GM1.124657；双FP32累加修订GM1.123799，仍慢约12.38%，16代表有2条改善>2%、11条回退>2%、3条在±2%、2条高CV。两套FP32累加使M16/32/64的独立MMAC目标向量2/4/8变4/8/16，但寄存器升至67/99/163；求和顺序改变仍通过原阈值全输出，静态链数增加尚未转化为当前最佳CK之上的整体收益。原96shape/53弱项属性规则拒绝，未抽新seed、未启动不合格候选的扩围／七block确认。

已证实的有限结论：K64减少SALU、B复用减少LDS指令／寄存器、N128降低LDS并提高驻留、取消A-load SLC、增加独立累加链，这些改动单独都不足以保证广泛收益。代码、资源、PMC和正式配对结果分别保留；具体动态等待／MMAC重叠仍需核对SQTT，不据高wait或单个长条宣布全部根因已解决。完整证据矩阵见readonly_reports/mechanism_matrix.md。

host修复只跳过不可能命中的优先selector；322845项CPU的status/instance/reason一致，全部1766个shader不变，35资格和10配对验证通过。更严格同进程11块CPU对照表明修复本身仅减少约0.04～0.16微秒；先前约1微秒是跨环境开关配置差值，不能全归因这次修复。GPU短控制仍有高CV，不计明确GPU收益。纠正及全部原始CPU记录保留。

原窗口已经完成H1/H2/H3、短K／低G两项原型和166条七block否定证据。本窗口续研348组三轮配对，旧个体28条稳定改善／14条归因跨桶因整域失败仍不能计入采用；原347计划中181项未完成、两个未配对精度和未测short确认维持历史状态，不能写成已完成。详见../final_closeout/closeout/README.md。

本窗口逐原始日志核对3187个作业：2976次全输出／25322607360个元素，精度失败0；没有待结束的GPU作业。另有1次profiler中应用rc134中止，原日志与失败状态保留并排除有效测量，不把它计作成功或已确认MKFD。第14批原8对只完成2对，部分包和稀疏尝试保留；第15/16批各4对完整完成。

所有本窗口GPU均物理4/epoch0/统一锁、fresh hy-smi及可读pre/post dmesg门禁。无新增环境恢复；累计445.379秒来自旧窗口dmesg权限事件，不能称为已确认硬件MKFD。没有reset驱动、换卡或跨epoch拼接。9套正式二进制和224份编译输入副本SHA均核对；构建包含受保护用户overlay，不宣称clean HEAD本身就是冻结测量二进制。

用户9处tracked贡献保留：7处改动块逐字不变；两处注册表数量随新增8个自有实例从61/63变69/71，用户原有+2贡献和条目逐字保留。10个untracked SHA及readfirstlane MD5保持，主Git index没有代为暂存用户改动。初始过严的数量逐字审计失败及纠正原样归档。

fresh_buckets_by_stage_variant_NOT_ADOPTED.csv只统计各阶段实际代表，全部未采用；每阶段逐shape及TFLOPS、配置、实际实例、轮次、CV、R_frozen、二进制/epoch和日志键在对应continuation_*归档。projection_P6k_UNCHANGED_NOT_FRESH1193.csv是零替换历史投影，不能称新鲜全量：prefill312为292/20/0/0/0/0，decode881为604/263/14/0/0/0。冻结BLAS来自GPU0 Native GroupGEMM，原同卡Native诊断单列R_live，不覆盖冻结值。

本窗口9个下载包共80份有效SQTT，逐成员SHA已核对。解压后先读XCOMPUTE_README，再开manifest指定perf、dispatch18和目标kernel/XCD/CU；第16批重点检查双累加寄存器及尾部FP32相加。perf时长不是正式成绩，不同wave累计时钟不能相加当墙钟。本机未运行Windows GUI；用户暂未查看第13批并要求继续，反馈不作为等待条件。

- 第9批：[P6l_batch9_phase_first_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch9_phase_first_SQTT.tar.gz)，10份；SHA256 `3314320967ae3e237f456f69f373c0a527ae577c25d7f1e274c39b638cf6f56c`。
- 第10批：[P6l_batch10_k64_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch10_k64_SQTT.tar.gz)，8份；SHA256 `733e699ff216ca4417a60da11356b9233ba1852a9026f8236b18091de00fe0bd`。
- 第10b批：[P6l_batch10b_k64_wide_depth_control_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch10b_k64_wide_depth_control_SQTT.tar.gz)，2份；SHA256 `9457a8885f96af87532e796f845fe4e6dee5bd002a01a34f41840d1366668bb7`。
- 第11批：[P6l_batch11_n256_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch11_n256_SQTT.tar.gz)，12份；SHA256 `dacf3c005fccff5b4c617cdc9649ea8c12f172711bb699e06eb1dc9a01a0831d`。
- 第12批：[P6l_batch12_b_reuse_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch12_b_reuse_SQTT.tar.gz)，14份；SHA256 `e49a20a859f54393ed86d68792aa56d75446062ab4bb26c632cea4678c71cd20`。
- 第13批：[P6l_batch13_low_lds_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch13_low_lds_SQTT.tar.gz)，14份；SHA256 `3be1c929cf353c347985a7b84ec9bff71b158b045e0f3a0cfb98df2a339ec0b7`。
- 第14批：[P6l_batch14_wave4_PARTIAL_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch14_wave4_PARTIAL_SQTT.tar.gz)，4份；SHA256 `a5f2d6532b2c223bf423b8267bea1ca7e98879d6dab2817bd1899ae6e637474d`。 **部分交付。**
- 第15批：[P6l_batch15_noslc_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch15_noslc_SQTT.tar.gz)，8份；SHA256 `6d71c58f480c3e8b97d039a83490d42bb68ec7798bcdd2abaf0ecfd5bce54156`。
- 第16批：[P6l_batch16_accum_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch16_accum_SQTT.tar.gz)，8份；SHA256 `c8b9a82d829ad17bd3bf6f0604244d09194ae799d1296f3c5df9e8cfd481a5ea`。

只读重建汇总命令：

`python3 -B /workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/readonly_reports/summarize_continuation.py`

`python3 -B /workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/readonly_reports/build_closeout.py`

原阶段控制器按冻结记录和实例核对执行，保留脚本原版本及各job键，不覆盖旧原始数据。当前新GPU门禁已关闭；这些汇总命令不会解锁、编译、恢复旧队列或发起GPU。后续实验须另有明确范围和有效期限，不能重写本窗口session来续跑。

全部提交仅本地、实验用、不合入；没有push。仍未实现广泛性能目标，后续工作应保留这些否定证据与真实数据口径。
