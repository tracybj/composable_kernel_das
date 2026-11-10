# 八级独立 Decode 流水接入及1193条统一复现

本轮在当前源码新增默认关闭的实例6003，使用A寄存器/B私有LDS八级流水。正式benchmark、默认selector和同一冻结计划完成全部1193条的全输出精度及各三轮CK性能；原P6g/P6i队列保持暂停。

## 新鲜全量分桶

|阶段|case|R≥0.99|[0.8,0.99)|[0.5,0.8)|[0.3,0.5)|[0.2,0.3)|R<0.2|
|---|---:|---:|---:|---:|---:|---:|---:|
|prefill|312|292|20|0|0|0|0|
|decode|881|604|263|14|0|0|0|

所有CK值均来自本轮，没有从历史表补齐未测项。R_frozen=原表冻结Native GroupedGemm BLAS耗时/本轮CK耗时；BLAS沿用旧卡号、repeat20协议和来源，未重测Native BLAS性能。逐组BLAS仅用于本轮全输出精度，不把该临时BLAS计时混入R。

原表prefill为293/19/0/0/0/0，decode为610/257/14/0/0/0。611/256/14是此前六条候选耗时替换的预测，不是本轮必须达到的成绩。实际高CV shape共有69条；全部保留质量标记，不据此声称稳定生产收益。

## 六条属性覆盖和ID1184

规则为gfx936、packed dense BF16 NN、common非空非ragged、M=1～32/N256/K512/G64。原表命中6条，其余1187条保持原开关和实例；另外26条合成shape只作范围验证，不增加正式case数。

|ID|M|原表CK ms|本轮新流水CK ms|原表R|本轮R|高CV|
|---|---:|---:|---:|---:|---:|---|
|1179|1|0.031599998|0.030239999|1.010633|1.056085|True|
|1180|2|0.031679999|0.030560000|1.010859|1.047906|False|
|1181|4|0.031840499|0.030400001|1.008778|1.056579|False|
|1182|8|0.031840000|0.030880000|1.024370|1.056216|False|
|1183|16|0.032960001|0.032000002|0.999998|1.029998|False|
|1184|32|0.034400001|0.033119999|0.973954|1.011594|False|

ID1184原表R=0.973954，本轮R=1.011594；是否跨过0.99以该实测值为准。新旧同二进制专项三轮对照另见qualification_results.jsonl，不能把与旧表的时间差直接当成本轮A/B收益。

所有跨桶见bin_crossings.json；all_positive_regressions.json保留全部相对历史耗时回退（含0～2%的不确定变化），all_regressions.json沿用冻结runner的>2%筛选口径。域外路径发生的变化属于原路径重新计时的观测，不归功于新流水；本轮未据此展开新的优化或根因研究。

旧路径复测的R≥0.99数量：decode从605变为598，净少7条（9条向下、2条向上）；新流水的六条从5条变为6条，净增1条，最终decode合计604。prefill旧路径有1条向下跨桶，合计292。共13条跨桶，其中12条为旧路径，1条为新流水；这些都是本轮新鲜观测，未按预测回填。
旧路径ID498耗时较原表增加13.56%（R 1.072585→0.944522），ID788增加5.95%（R 1.017533→0.960370），均无高CV标记；不能把这些较大差异一概解释为门槛抖动。新旧shader字节一致只证明旧计算代码保持，不能单凭本轮数据确定历史差异的根因；本轮没有擅自开展第二套全量或选择性重测。
全部相对历史变慢的点估计共786条，其中>2%共13条；其余0～2%仍保留不确定标记和全部原始数据。

## 同二进制六条A/B

|ID|旧配置ms|八级流水ms|耗时变化|存在样本高CV|
|---|---:|---:|---:|---|
|1179|0.032000002|0.030241000|-5.497%|False|
|1180|0.031680502|0.030239999|-4.547%|False|
|1181|0.032240000|0.030480001|-5.459%|False|
|1182|0.032320000|0.030719999|-4.950%|True|
|1183|0.033280000|0.031840000|-4.327%|True|
|1184|0.034639999|0.032880001|-5.081%|False|

六条同二进制A/B耗时点估计均下降4.33%～5.50%，其中2条存在单轮高CV。正式复测的ID1179也有高CV，ID1184靠近分桶门槛；保持默认关闭及实验性质，不把点估计或一次跨桶当成稳定生产采用。

## 精度与有效性

完整验收1193条（312 prefill、881 decode），1347次纳入统计的全输出比较、193,629,511,744个元素，failed=0；观测最大绝对差为0.0，未放宽原精度标准，CK/BLAS首尾真实guard均通过。
专项包含M=1～32全部整数各三次全输出、10条域外GPU回退、6条同二进制/同epoch新旧三轮A/B。原selector、Depth8 selector、query及smoke均通过，所有正式计时臂之前均有同shape的完整精度验证。
全部物理GPU4、统一锁串行、每作业fresh hy-smi/MKFD及可读内核日志门禁；warmup10/samples30/单launch/include_args_build0，精度阶段启用guard、计时阶段关闭其开销。每条精度及三轮属于同计划/二进制/配置/卡/epoch；三轮中位数再取中位数。
原始物理作业4866个，失败原始键[]，恢复记录{'seconds': 0, 'attempts': 0}。所有已启动作业均已终结，统一锁空闲。
高CV>5%、门槛附近和±2%变化保留不确定标记；不按快慢选样本、轮次或attempt。断点只接受最后一个完整有效attempt，异常窗口和旧记录保留。

## 实现和契约

新benchmark SHA256：76642f45892426cb97e758306f780d22c95567b38f9da475105aa38a0456b5ae。其原有54个控制shader与本轮开始的4ac20b…逐字节一致；新八级shader SHA：e252c83b232adef02ccbe90f2e009e601803d14bdef5928a15113ecd2e5ce925，与归档八级原型相同，动态LDS32KiB，零scratch/spill/CTA barrier。
新增源码/类型、ID6003、默认关闭ENABLE_BF16_NN_DECODE_DEPTH8开关及disable优先语义；同步CMake/provider/query/selector，原6000/6001/6002和暂停实验保持原含义，公开v1 ABI与GemmTransKernelArg未改。
存储验收为packed dense NN：lda=K、ldb=N、ldc=N及合法对齐地址。v1的host problem不含device kargs中的stride/地址，selector不回读它们；启用实验的调用方须符合该契约，不宣称支持未验收的任意stride。
复现入口隔离上一条和继承的所有CK_GROUPED实验开关，按只读源表派生冻结覆盖计划；CLI支持dry-run、pilot-only、qualification-only、resume和report-only。按表启用不同方案，不等于所有开关关闭时的一套生产默认策略。

## 实际命令和交付

实际执行命令及各阶段结果见commands.jsonl、build_command.json、build_result.json、cpu_tests.log；CPU dry-run和pilot记录保留。入口使用tools/grouped_gemm_depth8_reproduce/reproduce.py，具体说明见该目录README。

```bash
python3 -B -u /workspace/composable_kernel/tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan /workspace/composable_kernel/docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --output /workspace/v7_coverage/p6k_depth8_replay/run_20261110T060053Z --depth8 --resume
```

上式为本轮同期限断点命令；全量已完成时不新增测量。再次重新测量需用户新授权并使用新输出目录，不得在旧目录重置过期session。下面的一条命令会以时间和进程号创建新目录，自动完成CPU核对、专项、先导及全量；也可先追加--dry-run检查，再以--resume继续。
```bash
python3 -B -u /workspace/composable_kernel/tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan /workspace/composable_kernel/docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --output /workspace/v7_coverage/p6k_depth8_replay/new_$(date -u +%Y%m%dT%H%M%SZ)_$$ --depth8
```

results.csv/json为1193新鲜结果；frozen_plan.json和input.csv为只读输入/覆盖/身份；summary、跨桶、回退、所有失败和final_audit为审计。raw_logs.tar.gz包含全部实际命令/环境、健康和内核门禁、源脚本快照、精度/计时原始日志及异常窗口。二进制、原始code object只留现场，不放入Git。
源码与入口阶段本地提交：53d1417eec8a4a704a472fdc89f711d291d3f673（实验用，不合入）。原用户readfirstlane差异MD5仍4c7f49a5c7592767d53195a713db6b8a；原未提交P6i与tools未被代为提交。本轮不push、不进入下一轮优化。
从T0到生成报告已用7.052小时，T0=2026-11-10T06:00:53+00:00，截止UTC=2026-11-10T18:00:53+00:00。
