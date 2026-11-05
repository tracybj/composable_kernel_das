# INT8 NT操作数数据路径：LDS、单边LDS及寄存器

适用：gfx936、当前24_grouped_gemm_int8_nt、INT8累加INT32，kernel内反量化写BF16/FP16；静态B可预排一次复用。这里比较的是输入主循环，C epilogue仍可能使用LDS。v128为当前正式默认，v125–v133为隔离实验，不将实验支持误写为生产已启用。

## 数据路径的含义

GMEM→VGPR→LDS→VGPR并非经过两类物理寄存器。前一组VGPR按合并/16B搬运分工持有数据，后一组按MMAC片段分布持有数据；LDS负责跨lane/wave重排及共享。去掉LDS后必须解决这两种分布的匹配，且wave shuffle仅在同一个64-lane wave内交换，不能替代CTA内跨wave共享。

| 输入策略 | 能省什么 | 需要付出的代价 | 适合验证的场景 |
|---|---|---|---|
| A/B都经LDS | 跨wave复用，两侧连续搬运与MMAC布局可独立设计 | LDS读写、barrier和空间 | 较大M、有两侧复用、原始NT布局 |
| A经LDS，B寄存器 | 省B的LDS写读，A由各N方向consumer共享 | B需适配计算fragment，更多寄存器 | 原方向WASP已采用；中大M、packed B |
| A寄存器，B经LDS | 省A的LDS写读/空间，保留B的分发与重排 | A在N方向waves重复读取 | 小M、A量很少；紧凑XOR B的对照 |
| A/B均寄存器，必要时shuffle | 省输入LDS读写及主循环CTA同步 | 重复加载、shuffle、VGPR及非连续访存可能增加 | 小M、复用较少、物理布局匹配 |
| 原始A+预排B，均寄存器 | B合并读取；同一K排列可免shuffle | B一次预排与额外存储；只适用于可复用权重 | 小M专用的Cooperative实例 |

原方向WASP的A producer→双缓冲LDS、B consumer→寄存器，本身就是单操作数LDS方案。逻辑转置后角色按内部数学操作数交换，不能一概称为原始A始终进LDS。

## 已完成的可复现证据

1. v127相同BM16/BN128/BK256对照：M32/N2048/K512/G64，双LDS23.43 TOPS，单边LDS+延后shuffle约36.06 TOPS；但当前另一tile的默认41.89更快。局部数据路径收益不等于超过全局当前最优。
2. v130只B紧凑LDS将BM16/BN32/BK512的输入占用压到16KiB，HIP静态最大驻留2→4CTA；BN128/BK256静态1→2CTA。长K小M整体没有全部达标，资源减少本身不是目标。576项完整范围数值门禁及608项正式输出检查PASS。
3. v131 M1/N4096/K6144/G8，相同BM16/BN32/BK512，原始B的双寄存器16B直取1.7902 TOPS，预排B双寄存器2.3093 TOPS；原默认2.2246 TOPS。B的物理供给模式是这个组合的重要条件。522项full-range门禁、760项正式完整输出检查PASS。
4. v132同一shape，FullNK预排寄存器候选约2.347 TOPS，超过目标约2.7%；整块N/K约束允许去判断，M仍mask。19组418项完整输出检查PASS。其它小M仍未全部超过目标，不能统一替换。

各版本同轮对照分别见max_all_v130_compact_formal.csv、max_all_v131_packed_formal.csv、max_all_v132_fullnk_formal.csv，完整证据及编译SHA见MAX_ALL_OPTIMIZATION_LOG.md。上述数字不跨版本拼接作增量因果比较。

## 继续迭代的判据

保留Cooperative与WASP两套核心家族，把ALds/BLds、fragment重排、是否prepared B、tile和预取作为实例策略。生产分流按M/N/K/G、stride、dtype和pack合法性，不用case ID。输出格式和scale语义相同，原无scale INT32接口保持。

先在相同tile/wave布局下比较双LDS、单LDS、纯寄存器，说明哪些布局改动必须联动；再与当前默认最优比较。记录实际VMEM/交换指令、VGPR、LDS、spill、静态可驻留CTA、完整时间；不能仅凭LDS小或occupancy高推广。候选先通过完整范围INT8/FP64点积、BF16/FP16舍入及特殊scale、tail/padding和guard验证，再做独立正式复测。权重一次预排时间/存储与GEMM稳态分别报告；不可把动态A准备移出计时。

v133在预排纯寄存器输入中进一步比较32位相对字节地址：基地址仍为64位，并受API矩阵span约束。当前仅实验，结果另记优化日志；无稳定收益则保留已有路径，不以省指令推断吞吐。

## v133实际结果补充

32位相对offset的四种实例均通过full-range门禁，19组342项正式完整输出检查PASS。VGPR由156/128/126/240降为145/119/119/221，第一种静态最大CTA由8增为10；但同几何配对增益中位数仅约0.03%–0.44%，未形成统一切换的依据。地址指令、寄存器、静态驻留这些观察均成立，仍不能替代完整性能结论。数据见max_all_v133_address32_formal.csv；默认生产分流保持v128。

单操作数进LDS是数据路径策略，不能据此将Cooperative称为WASP；WASP还包含独立producer/consumer角色及发布/退役同步协议。也不能把输入不进LDS称为整个kernel无LDS，输出重排仍保留共享内存。
