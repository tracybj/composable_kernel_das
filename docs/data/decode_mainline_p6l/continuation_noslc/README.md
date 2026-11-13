# N128 四 wave：A 加载缓存标志的受控修订

保持 tile、wave 分工、LDS 深度、三个操作数缓冲和 selector 不变，仅移除实验实例 6035 的 A-load 显式 SLC 修饰。正式配对的两臂均来自本轮同一冻结二进制、GPU4、epoch0；历史6035耗时只作跨构建诊断，不能拼入本轮成绩。

benchmark SHA256：`ed5631780a58479627d9b5c32324322eba72ecdaaa91ef5ee48993aad1c6d11b`。全部35资格作业、23个三轮配对、237次全输出比较及真实guard通过。

16条实际代表的耗时GM（候选/当前CK）为 1.124657；确定三轮均回退超过2%的ID：[805, 312, 327, 530, 543, 712, 933, 1170, 686, 687]。固定属性域96条原shape/53条弱项被拒绝，未抽取新盲测seed，也未启动七block独立确认。

CPU六个双目标探针资源/大小不变；正式构建保留1760个非6035 shader字节，6035六个shader改动且仍零spill/scratch，三个实际CFG审计通过。另对正式二进制gfx936的三段完整指令体逐指令核对：相对PC、操作数和编码除各14条A-load的SLC位0x20000外均相同。不能把这些静态事实称为性能收益。

已交付 8 份有效SQTT。XCompute说明及manifest在profile_delivery/；本机核对完整性和实际kernel/dispatch/代码，GUI未运行。不同wave的累计时钟不能相加当kernel墙钟。

paired_results_with_tflops.csv保留所有正向、回退、高CV、配置、实例、轮次、TFLOPS、R_frozen与原始日志键。冻结BLAS来自P6k历史GPU0 Native GroupGEMM，不冒充本轮GPU4新鲜BLAS。

没有新采用方案，50/30目标计数仍为0/0；P6k、1193shape与原Excel保持不变。只做本地实验提交，不push、不合入；原用户修改保留。大型perf、二进制、厂商ISA和用户overlay源码不入Git，实际26份编译输入及SHA保留在现场compile/build15。
