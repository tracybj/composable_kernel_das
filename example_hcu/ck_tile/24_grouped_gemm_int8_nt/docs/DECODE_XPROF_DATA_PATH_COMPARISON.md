# 小M四路径同条件xprof对照（2026-11-06）

M1/N4096/K6144/G8，INT8 NT→BF16，FP32 scalar A × per-channel B。卡7，库v143（SHA6770132c...446ff3b），固定BM16/BN32/BK512、128线程，输入/scale四组SHA一致。每份捕获仅一个完整G8 grouped kernel，grid为全局线程(16384,1,8)、workgroup(128,1,1)，不是只看一个专家。所有完整输出逐bit及padding guard通过；FP64整数点积→INT32→FP32组合factor→BF16 RNE参考。

## 无profiler独立计时

同一进程准备四种plan，预排静态B只做一次；warm10/repeat50、平衡轮换顺序、3样本中位数。仅kernel时间，预排不计入可复用权重稳态。

|路径|route/instance|ms|TOPS|VGPR|LDS bytes|
|---|---|---:|---:|---:|---:|
|A/B双LDS|0/8734|0.179917|2.2380|116|25600|
|A Reg、B CTA LDS|0/9041|0.185456|2.1712|160|16384|
|Raw A/B双Reg、共同K64分布|0/9054|0.224243|1.7956|140|2048|
|Packed B双Reg、共同K64分布|32|0.173856|2.3160|156|2048|

LDS为工具报告实际分配（含分配粒度），不是源码公式取整前的字节数。2048仍供C epilogue交换，双Reg不等于整个kernel没有LDS。资源数不是实际occupancy证明。

## 报告观察

|工具counter|Raw双Reg|Packed双Reg|
|---|---:|---:|
|SQ_INSTS_VMEM|399360|399360|
|SQ_INSTS_LDS|14336|14336|
|TCP_TOTAL_READ_sum|25296896|25296896|
|TCP_TOTAL_CACHE_ACCESSES_sum|13405696|3968512|
|TCP_READ_TAGCONFLICT_STALL_CYCLES_sum|9437184|0|
|TCP_TCP_TA_ADDR_STALL_CYCLES_sum|10201760|318493|
|TCP_TCC_READ_REQ_sum|3268651|3282653|
|TCC_MISS_sum|3150444|1577223|
|TCC_EA_RDREQ_sum|3149305|3148970|

Raw的TCP cache access约3.378倍；读tag冲突stall计数与两者cache access的差在本捕获中数值相等。两者TA FLAT read wavefront、TCP total read和MMAC工作量相同，下游TCC读请求接近。这个证据支持优先研究原始NK步长引起的片上tag冲突/请求停顿；不能简单说T0/T16不能合并，也不能把TCP access当唯一cache line数或HBM字节数。

TCP冲突的具体set/bank映射、重试机制仍待控制实验；计数差数值相等不是已经核实的公式。TA_BUFFER_*均为0，本次走FLAT，不能用这些0推导访存合并率。不能把跨CU/wave的stall总和换算为kernel壁钟百分比。

当前安装gfx_metrics.xml的gfx936继承gfx9；SQ_WAVES/SQ_INSTS_VALU/SQ_INSTS_LDS等明确标为emulated。这里只保留工具计数与相同工作量检查，不把它们当独立硬件指令计数证据。TCC读请求和UMC引擎全覆盖未核实，不报HBM物理带宽或roofline百分比。解析器仅抽取经过布局断言的uint64记录，不生成UI派生百分比。

四份.perf与完整SQTT原始文件在容器results/decode_xprof_v144，原始清单/哈希见decode_xprof_v144_artifacts.json；本地副本在D:/yyc_workspace/GroupGemm/analysis/decode_xprof_review_20261106/capture_v143。工具/metric定义、精确命令、单dispatch元数据、完整输出校验、无profiler原始样本均保留。

复核：experiments/operand_paths_v125/parse_xprof_counters.py <本地capture目录>。Raw/Packed同几何但寄存器数不同，不把所有性能差全部归因一个counter；后续考虑同地址模式下减少跨行并发、修改copy映射或静态权重布局的控制实验。
