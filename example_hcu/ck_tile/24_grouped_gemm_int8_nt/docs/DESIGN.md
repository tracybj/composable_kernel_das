# INT8 NT Grouped GEMM 设计与验收口径

2026-10-26；原始源码 revision：`9acdc34e1f733f3c5dbfc85b52d6273409ec2ce8`。

## 范围

本轮用户确认采用有符号 INT8×INT8→INT32，无 scale，alpha=1、beta=0；NT物理输入 A[E,M,K]、B[E,N,K]，输出 C[E,M,N]。量化INT8输出和per-channel scale后续补齐。在ck_yyc、物理卡7、gfx936上验证，保持每组独立权重和完整E/M/N/K。

需求附录1178条：decode872、prefill306，26组K/N。需求SHA256：`5e62afd433a62b56b3e5861e0f61aeed771bce91b9a4e00ac35042d647f04b4b`。逐条目标为正确且 R>=0.5，R=CK TOPS/BLAS TOPS=BLAS ms/CK ms。历史基线达到目标263条，其中prefill为0。代表用例与全部结果分别统计。

## 实现

小M路径以K32的连续8字节直接装入CK现有MMAC attribute的打包片段；每wave16×16，4wave组成16×64 CTA，不使用LDS。

其余路径把搬运分布和计算分布分开：每线程16字节全局加载，保留打包VGPR，再写入行距BK+16的LDS；wave按MMAC片段读取，使用原生INT8 MMAC和INT32累加。K64单缓冲流水预取下一tile，所有wave读取完成后才覆盖LDS。peeled版本将末尾计算移出热循环。编译调度屏障不替代LDS同步。

输出通过LDS重排后按16字节写回。小K候选合并多wave输出条带，使用可逆列swizzle；大K候选保留较小暂存。大tile将设备参数先复制为局部值，避免K循环反复读取参数。每个CTA负责独立输出，无split-K、原子累加或输出中间结果。

代码按problem、policy、pipeline、epilogue、kernel分层，instances负责预编译实例和launch；direct小M路径独立维护。通过原公开select/query/run v1和GemmTransKernelArg接入默认selector，只兼容扩展dtype、实例ID、mask及reason。更新provider清单、CMake及原统一C++ benchmark。选核仅依赖架构、算子属性和对齐，不按模型名或phase选核。

无边界检查的快路径只接收common且无ragged/empty标志、完整tile对齐的输入，其余使用mask。使用调用方stream，run不分配、不回读GPU长度、不做设备同步。新dtype首版要求num_cu=0；非零在launch前报unsupported。FP16/BF16旧路径保留并运行原80条selector回归。

## 验证

固定BLAS安装 `/workspace/GroupGemm/baseline/hipblaslt-install`，核对实际加载路径、SHA和算法身份。复用历史逐shape遍历有效算法、top3精调后的最快有效选择，并在同轮重新计时；ID1110使用此前补测恢复的算法。

正式一轮，每样本预热10次、计时50次，三个HIP event样本取中位数。`--screen`使用更短计时对照历史BLAS，仅为筛选，BLAS状态明确标记HISTORICAL_PASS，不是正式同轮验收。

Python逐组全输出参考使用[-8,8]输入；需求K范围内所有整数部分和可被FP32精确表示。检查完整输出、组前后guard、输入抽样和expected-instance拒绝。额外边界测试使用[-128,127]完整输入范围、FP64全输出参考，覆盖ragged/empty、stride/tail、非默认stream及错误参数，并完整比较输入副本和输出padding。统一C++入口另做CPU精确抽样与全量CK/BLAS输出比较；数值PASS与R达标分别报告。

逐条保留phase、M/N/K/E、实例、两库ms/TOPS、R、样本、算法身份、库和源码SHA。续跑要求二进制和计时manifest一致；不覆盖完成记录，不对整权重一次性转INT64。测试进程关闭core输出，不改全局环境。

## 最终状态（2026-10-27交付核对）

1178/1178条正式数值通过，1178/1178条R>=0.5；min R=0.508321，总体中位R=1.020894。prefill306条、decode872条分别全部达标。未跳过显存不足的shape，无缺失指标。

原80条及新增20条selector检查、13组完整范围边界、C11 ABI布局检查通过；原615行FP16/BF16命令构造逐条相同。统一C++入口的全1178条time=0数值验证也全部通过，代码bundle与正式测试库一致。其单次验证时间不计入正式性能。

正式CSV、原始记录和审计位于results/final_01，统一入口证据位于results/unified_validation_1178；docs/FINAL_REPORT.md和PIPELINES.md给出汇总及实例统计。当前仅交付gfx936 INT8 NT、INT32输出、无scale；量化输出和scale继续按已确认范围后续补齐。
