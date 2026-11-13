# P6l 收尾报告：性能目标未达成

**本轮没有可采用的新方案。50条稳定改善/30条可归因跨桶的完成数均为0；原P6k正式表、冻结BLAS和1193 shape不更新。**

首轮独立确认完成166条原表主线，耗时几何平均比为1.053153（约慢5.32%），105条中位回退超过2%，其中94条低CV且七个block均回退超过2%。因此整个域拒绝。原347配对中181项未完成，两个未配对精度作业保留；short确认未测，不能称为成功或失败。

原表弱项中，个体满足七block稳定规则的有28条、个体满足可归因跨桶的有14条；**因整域失败，两者都不计入Goal，也未回填到正式表。**

## 研究与验证范围

| 阶段 | 完整发现配对 | 结论 |
|---|---:|---|
| H1：A寄存器/B私有LDS、完整M |48| 主线及短N路径未获得整域收益 |
| H2：A共享LDS/B私有LDS |39| 部分代表获益，仍有回退 |
| H2：布局与预取顺序因素 |74| 有局部改善，未采用 |
| H3：A/B协作LDS、两层预取 |85| K32代表有收益，不能外推整域 |
| H3：深度及host解析因素 |116| 代表改善未通过后续166条确认 |
| H3：A2/B3四wave、D4八wave资源因素 |46| GM分别1.018896/1.032618，均有明确回退 |

总计408个三轮配对，另有166个七block独立确认配对。它们有重叠shape、多个构建和epoch，不能当作574个独特shape或拼成逐shape历史最快表。完整变体结果见research_variants.csv。

主线研究使用三种所有权结构；K16/K32、深度、布局、资源及调度变化是实例/因素，不是额外的所有权架构。短线已做原型和三轮实验，得到负面结论；未得到采用方案。

## 最后调度原型：仅编译/静态证据

当前正式路径二进制为 `ee64d307fb6f506ff84326b3b2537351aee0c13ca1b1cce89604ebc302454d1d`。六目标构建于2026-11-12 17:57:39 UTC结束；已有1710个shader机器码保持，新增10个shader零spill/scratch，gfx936五个循环/寄存器依赖审计通过。宽wave M32/M64实际VGPR为73/112，四wave为63/86/133。

**没有用这个新二进制跑GPU query/smoke、全输出精度、三轮性能或SQTT，不能把寄存器减少当作性能收益。** 计划中的27项资格、46个配对、profile9以及新的完整属性域独立确认均未执行。新开关默认关闭；该代码是未验收实验。

静态设计为：先计算当前MMAC，再执行原VM/LG等待和CTA barrier，然后读取未来操作数、预取后续A/B。每wave仍完整累加K，收尾的最后ready(0)发生在该块MMAC之后、Store C之前。移植源中一条旧compact收尾注释仍写before computing，应以代码和实际CFG审计为准；为保持构建输入SHA，本轮收尾不修改源码。

## 数据口径

measured_rejected_confirmation166.csv包含本轮同二进制/epoch配对数据、TFLOPS、冻结BLAS比值和全部七block；候选明确标记REJECTED。其分桶仅代表这166条，不是新全量。

projection_P6k_UNCHANGED_NOT_FRESH1193.csv是原P6k未替换的历史投影，包含TFLOPS及逐行来源；不是本轮1193复测。互斥桶如下：

| 阶段 | 总数 | R≥0.99 | 0.8≤R<0.99 | 0.5≤R<0.8 | 0.3≤R<0.5 | 0.2≤R<0.3 | R<0.2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill |312|292|20|0|0|0|0|
| decode |881|604|263|14|0|0|0|

R使用P6k冻结的Native BLASLt GroupGEMM耗时。16条同卡Native诊断的R_live另存，未覆盖冻结BLAS。以上是任务指定P6k基线，不是更早P6f截图中的610条桶。

## 数据与环境审计

closeout/audit.json逐原始日志核对了6632个作业：CK全输出4830次/17873304576个元素，Native全输出96次/270385152个元素，失败0。记录覆盖186个原表ID，重复作业不等于独特shape。所有作业为物理GPU4，健康门禁、pre/post dmesg、默认selector实际实例、原始轮次和版本记录均通过核对。

环境恢复1次、累计445.379秒，由dmesg权限不足触发，并非已确认的MKFD硬件故障。独立新进程按两次健康观察及固定5000探针恢复；epoch0/1分别保留，不重启驱动、不换卡，数据不跨epoch拼接。新原型没有GPU记录；全部已启动GPU作业已有终态，宿主进程名称快照另存process_snapshot.txt。未运行1193或881全量GPU测试。

## 截止与未完成原因

T0为11月12日02:53:23 UTC；新优化/编译/GPU提交截止为当日22:53:23 UTC，硬截止为11月13日02:53:23 UTC。等待、编译、中断和恢复全部计时。构建完成后接续GPU验证没有启动；恢复时已超过提交截止，所以没有补交实验或重置时钟。这是本轮未完成项，不能归因于等待用户确认，也不能伪称已验收。

后续只能在用户另行授权的新窗口中验证候选。旧session、失败记录和原验收门槛必须保留；当前继续按钮不会延长本轮期限。已准备的入口和计划不等于已跑通的一键验证，需在新会话重新检查并冻结。

## Profiler / XCompute 文件

共8批包，已重新核对大小、SHA、tar路径、README和manifest。本机未运行Windows XCompute，GUI表现仍需用户实际查看。 Source指令锚点、dispatch/XCD/CU和Wavefronts相对窗口见各包XCOMPUTE_README.md；perf时长不是正式采用成绩。

- 第1批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch1_CK_Native_SQTT.tar.gz)；173174075字节；SHA256 `2f9967942eac2b663397fb76080e4a82e457f9af3f7a26187b4fc28c66ad6788`。
- 第2批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch2_H1_current_candidate_SQTT.tar.gz)；2756602字节；SHA256 `7b82410cd09a7bb89d21f8d23d697f571906f1ee1df29bc060cffc651f30e297`。
- 第3批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch3_H2_current_candidate_SQTT.tar.gz)；6332899字节；SHA256 `4d867d598f11b4d69cc1671151d920cfb7b2509d1a389973941b0375129ea4cd`。
- 第4批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch4_tuning_current_candidate_SQTT.tar.gz)；3099686字节；SHA256 `c96c73b86a2b2bc4a0fc41e231737d6f73a00883dcb65516a600809bcdf03a8d`。
- 第5批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch5_h3_current_candidate_SQTT.tar.gz)；3922229字节；SHA256 `93c01cd3bd357007cc5365fccd2c87facee2c345fd2a74c7a4f545b4765c0048`。
- 第6批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch6_h3_factors_current_candidate_SQTT.tar.gz)；10188506字节；SHA256 `429cf850856d86aab0263a3041b102b1d2c9357be6d57b428bf8903f01a2e247`。
- 第7批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch7_h3_regression_SQTT.tar.gz)；3973344字节；SHA256 `629f381331866b13608513a1351c9b9d02c20a81a60f624e65cabbd3fb768981`。
- 第8批：[下载](/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch8_resource_factors_SQTT.tar.gz)；4696642字节；SHA256 `3b79cf78bb28d5a14710f4beed7083d49ad3526b890ff39dd7e9a930022afe89`。

所有提交仅本地、实验用、不合入；原用户tracked/untracked改动保留。性能目标保持未完成。
