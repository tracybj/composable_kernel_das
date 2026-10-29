# DeepGEMM gfx938 FP8 对当前 gfx936 INT8 GroupGemm 的借鉴分析

分析对象：用户提供的 `/root/workspace/yyc/GroupGemm/deepgemm/csrc/arch/gfx938`。本次读取的 DeepGEMM revision 为 `fc246fc4e4104c55a16b88d2f5682be54690469c`。测试仍在 ck_yyc 的物理卡7、gfx936 上进行，使用当前 DTK。没有把 FP8 在 gfx938 上的结果作为 INT8 在 gfx936 上的成绩。

## 1. 找到的实现分支

| 分支 | CTA / wave 划分 | 主要数据路径 | 与当前接口的区别 |
|---|---|---|---|
| contiguous FP8，BM128/BN256/BK64 | 4个wave；每个wave负责128×64；8个M16×4个N16累加片 | A双缓冲LDS共16KB；B预排后直接进入寄存器；A寄存器滚动预取 | gfx938的matrix-load和FP8 MMAC；预排权重；FP32累加、scale、BF16输出 |
| contiguous FP8，BM64/BN256/BK128 | 4个wave；每个wave负责64×64；4个M16×4个N16累加片 | A双缓冲LDS共16KB；每个BK128含两个K64；B预排并预取 | 同样依赖预排与FP8/gfx938路径；NLoopNum=2仍对应4个N16片，不能把参数名直接当最终N片数 |
| pair32 WASP FP8，BM256/BN256/BK128 | 4个producer+8个consumer；后者分C/C′两种时序 | A双缓冲LDS共64KB；B预排后直接进入寄存器；C′跨stage持有后半A计算 | 基本组织与现有WASP接近；C′不是当前INT8上已经证明有效的配置 |

此外，共享目录中发现 `m_grouped_w8a8_gemm_nt_contiguous_gfx936.cu`，静态代码同样是128×256×64、4-wave、预排B，并包含INT8 MMAC分支。但当前setup.py对该源文件有gfx938目标限定，所以不能仅凭文件名认定已经在gfx936编译或实测通过。共享目录另有INT8 pair32入口。gfx938目录的另一个i8入口引用的dispatch目录在本次checkout中缺失，不能直接当成完整可运行实现。

## 2. 真正可迁移的内容

**第一是任务划分。** 每个wave一次持有多个N16片，让一份A寄存器参与更多计算；小一些的CTA加上较少wave，可能减少LDS和每个CTA的总寄存器需求。但B会被更多M方向CTA重复读取，资源下降不保证速度提高。

**第二是B的物理布局。** 普通NT输入是`B[N,K]`。其一组lane分别读取多行中的16字节，物理地址跨越K行距。参考代码先把权重排成`[E,K/64,N/16,4,16,16]`（pair32还有独立布局），让一个N16×K64 MMAC片连续存放。逻辑矩阵乘不变，但输入准备、存储布局和接口发生了变化，必须明确转换是否可摊销。

**第三是生命周期和生成代码。** A的LDS双缓冲、A寄存器滚动、B预取深度、C/C′时序需要共同设计。当前INT8 MMAC的寄存器表示、编译器等待插入和gfx938 FP8并不完全一致，不能只替换一条MMAC指令后认定流程等价。

参考实现中的A scale与B scale、BF16输出属于它自己的数值契约。本轮仍为INT8×INT8→INT32、不含scale；没有以降精度或改变累加契约换取速度。

## 3. 已完成的隔离实验

### v87/v88：结束原有预取微调

v87提前B预取的28对同映射/同record比较全部变慢。反汇编发现未来B加载被额外生成的首条MMAC前VMEM等待收住，未形成预期重叠。v88用编译器可见的wait builtin移除了该额外等待，480项全范围数值检查、112项性能输出/guard检查通过，但最优仍是原来的late-prefetch对照，没有可推广的稳定提升。详见对应迭代文档。

### v89：4-wave任务划分，保持原始NT输入

独立实现CK-Tile Problem/Pipeline/Kernel/Epilogue，128×256×64、4-wave协同搬运/计算，A读批量1/4、record0/1024、两种内部映射。保留现有INT8寄存器表示和LDS搬运，未复制DeepGEMM源代码，也未使用gfx938专属matrix-load，因此是任务划分的隔离实验，并非对其完整FP8流水的复刻。

248项全范围INT8/FP64数值检查、84项代表shape同轮输出/guard检查全部通过；180/204VGPR、16KB LDS、无spill。但7个代表shape均未超过同构建WASP对照。按相同256×256输出、K128归一，名义A+B请求由64KB增加到96KB；这是请求量估算，不能当成实测HBM流量。详见`iteration_89_n_reuse.md`。

### v90：只改变B物理布局，保持现有256×256 WASP计算主干

本次实验沿用256×256×128、4producer+8consumer、统一C时序和现有MMAC算术，新增K64/N16预排B读取；raw record0/1024作为同构建对照。按内部原方向预排的是原始权重B；内部转置方向预排的是原始激活A，后者通常每次请求变化，不能直接按静态权重摊销。

96项全范围INT8/FP64检查通过，包含多stage奇偶、padding stride、多组、空组及诊断fallback；输入完整克隆和输出guard检查通过。预排分支198VGPR、64KB LDS、无spill；raw对照214VGPR。42项代表shape输出/guard检查通过后，用同一批内存分配、4候选轮换顺序作性能复核。完整数值和时间见`iteration_90_packed_b.md/json/csv`。

## 4. 口径与下一步

预排的GEMM时间与GPU转换时间单独记录。本次转换使用预分配输出的PyTorch布局拷贝，是正确性参考实现，不声称已优化到转换带宽上限。转换时间不含输出分配；“一次使用时间”是转换与GEMM各自中位数的分项相加，不是完整API端到端实测。

原方向可研究在专家权重加载时预排一次并缓存；需要明确权重更新/失效规则、是否保留原始副本、额外存储和新的packed-weight接口。原始NT接口继续保留。转置方向若预排激活，必须把每次转换计入调用，或另行验证确实存在的跨算子复用，不能套用静态权重假设。

若验收必须直接接受原始NT输入且所有准备都计时，下一步应围绕B访问组织、片上重排与计算融合继续研究，不能把预排后的核吞吐登记为原始接口的达标数据。现有正式v80的1178条CSV和默认选择器保持不变：975条达到冻结BLAS目标，203条仍待提升；全部超过目标50%。这里没有新一轮1178条正式验收，也不作“已经全部超过BLAS”的结论。

## 源码证据（仅路径/摘要，不纳入参考源码副本）

- `gfx938/m_grouped_fp8_gemm_nt_contiguous/bm128_bn256_bk64_nloop4.inc.cuh`：kernel模板、4-wave/4N片约208–250行；A/B装载与滚动计算约330–480行；6D权重与launch约674–735行。SHA256 `4c70b703e7d349f2eb1bacf2f26801cd040370c9ab8246da4d5302b308343938`。
- `gfx938/m_grouped_fp8_gemm_nt_contiguous/bm64_bn256_bk128_nloop2.inc.cuh`：参数/资源约354–400行；分两段K64、B预取约450–570行。SHA256 `edf7e79856e200e3bb16621bdfb82c8a1432e3c3e30d1823b77521c3244ef816`。
- `gfx938/m_grouped_fp8_gemm_nt_contiguous_wasp_pair32/bm256_bn256_nloop4.inc.cuh`：线程/资源约62–79行；C/C′时序约661–754行；7D权重检查约847–880行。SHA256 `d198cb1ed89c66a506c14bb96d29e0df5dd08784d1b0649cd16a2c1617087c2b`。
- `gfx936_gfx938_shared/m_grouped_w8a8_gemm_nt_contiguous_gfx936.cu`：INT8 builtin条件约182–187行；任务划分/寄存器流水注释约215–243行；两stage A加三组B约400–540行。SHA256 `8072b9dddf56b033b6dca0d575ca7b471d89c9d58f69f2cfea79fe7e5f11110c`。
- 本地必要参考副本保存在分析工作区`deepgemm_gfx938_reference`，不复制到自有代码GitHub仓库。新候选基于现有CK/自有VMEM与MMAC助手独立实现。

## 后续隔离验证：v91原始NT下A/B均进LDS

256×256×64、8-wave、A/B双缓冲共64KB LDS，消除了外部预排；124项全范围与56项代表输出检查均通过，151/175VGPR、无spill。但七组均慢于同轮WASP控制。反汇编确认当前计算与下一stage预取交错，首MMAC前无额外VMEM等待，不能沿用v87的失效解释。详见`iteration_91_lds_operands.md/json/csv`。正式默认与1178条结果保持v80。
