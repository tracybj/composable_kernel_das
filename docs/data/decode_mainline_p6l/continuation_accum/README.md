# 固定 N128/K32/D2/W4：两套 FP32 累加缓冲

同一K32块的两个K16部分分开累加到C0/C1，收尾逐元素相加。完整M所有权、A/B共享LDS、三套操作数缓冲、加载/读出/存储函数及selector不变；这属于既有H3的累加链因素，不是第四种所有权架构。未采用实验6035按源码/二进制SHA区分实现修订，6000/6001/6002/6003及1760个其他shader均保持。

benchmark SHA256：`dd8e20c51f611150ada550ac634d68ff102237ee96fbba5e7dc4a590d1fc37ad`。35资格、23个三轮配对、237次全输出比较及真实guard通过；求和顺序改变，精度阈值未放宽。

正式gfx936主循环的独立FP32 MMAC目标向量数，在M16/32/64分别从2/4/8变为4/8/16，静态MMAC总量不变；VGPR为67/99/163，6个双目标shader无spill/scratch。独立链增加是否改善实际重叠仍需SQTT和正式计时，不能单凭静态数量宣布收益。

16条实际代表GM（候选/当前CK）为 1.123799；三轮低CV且全部回退>2%的ID为 [805, 312, 530, 543, 712, 933, 1170, 686, 687]。代表发现仍有确定回退，固定属性规则拒绝。

属性域仍为96个原shape/53个原弱项。没有按赢家ID过滤，没有抽取新盲测seed，没有七block采用成绩；50/30目标计数仍为0/0，P6k和冻结BLAS不更新。

第16批交付8份有效SQTT，目标dispatch18/实际kernel/代码均核对；GUI未在本机运行。先读profile_delivery/XCOMPUTE_README.md，查看MMAC目标寄存器、DS read/等待重叠与尾部FP32相加。不同wave累计时钟不能相加当kernel墙钟。

所有正式两臂均来自本轮同一冻结二进制、GPU4、epoch0。历史单累加6035跨构建值仅为诊断，不混入本轮成绩。paired_results_with_tflops.csv保留全部正向、回退、高CV、配置、实际实例、轮次、TFLOPS和R_frozen；原BLAS来自冻结GPU0 Native GroupGEMM。

原1193shape和原Excel保持不变，只做本地实验提交，不push、不合入；用户修改保留。大型perf、二进制、厂商ISA和用户overlay源码不入Git，实际26份编译输入保留现场compile/build17。
