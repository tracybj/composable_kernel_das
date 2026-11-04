# P6d: Native GroupedGemm ISA-guided diagnosis

用户目标为全部1193个shape接近或超越冻结BLAS参考，沿用历史R=BLAS/CK和0.99分桶边界；本阶段完成反汇编及代表shape验证后按工作区阶段规则汇报。没有把“接近”新定义为一个更宽松阈值。

- 物理卡4串行。每条GPU命令前hy-smi确认HCU%=0；忙则停止，不换卡。原始日志、所有轮次和失败保留，run.lock防止自身并发。
- 不做1193全量。九条代表：76/633/681（低CTA），358（对照），146（N1232只读诊断），772（小M回退），1186/1187（新增M128/256），1178（大M大K）。
- 精确复用冻结Native GroupedGemm算法ID与名称，不重调BLAS；桥接的Native M=N、N=M，ISA的MT需交换前两维后再和CK比较。
- CK使用合规benchmark默认selector，分别关闭/开启既有small-M128实验开关。没有强制kernel ID，没有修改5000/5002/5003。warmup10/samples30，每样本单launch。
- Native在同一进程分别测repeat1和repeat20，各30样本；三轮交替CK原路径→small128→BLAS。repeat20用来核对历史计时差异，不把它与CK单次launch的差距全部归为shader损失。
- 第一轮8条中小shape的Native输出做CPU FP32完整参考+guard；1178避免重复巨型CPU GEMM，沿用冻结基线已通过的精度证据。CK所有计时继续CPU抽样+guard；本阶段没有新kernel，未把抽样描述为全元素验证。
- profiler补146、772、1186、1187、1178共13组配置，其余4条可复用P6a/P6b已归档计数。每组warmup1/samples1，保留两个dispatch，核对MMOP/waves几何。
- SQTT先对681/5003做4MB、单CU、replay off探针；只有输出有可解析时序信息时才扩大。静态指令数含不执行的尾块/activation/epilogue分支，不是动态指令或耗时。
- 原始BLAS机器码、反汇编、code object仅存本地p6d/。Git只保存自己的提取/分析脚本、哈希、派生指标、报告和小型运行证据；不复制BLAS实现到CK。
- 保留未提交readfirstlane改动，不stage。N1232尾块实现继续遵守用户“在决定前不要动”的边界。本阶段不变更公开ABI、selector和默认策略。
