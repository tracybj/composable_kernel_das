# ID76：A-LDS/B-register 原型与旧 K32 的 SQTT 对照

2026-11-09，仅物理 GPU4 串行；M=1、N=2048、K=768、G=4。

下载文件：
- `ID76_A_LDS_B_REG_SQTT.perf`：trial06b 新原型，实例6000。
- `ID76_baseline_K32_SQTT.perf`：既有小M K32，实例5008。

用 XCompute 打开新的文件，选择 dispatch18、XCD0。通过 Wavefront/Inst 查看所采 wave 的指令时间线，通过 Source 查看对应的指令、Hit Count 和 Latency（以 GUI 实际解码结果为准）。两份文件均含原8类 section的 PMC 和 SQTT；详情参见 summary.json。不要使用此前不带 SQTT 的旧文件。

采样限定 SE0 的首个活动 CU，文件硬件编号为CU1；不能把该CU的指令时间线当成所有CU的全局耗时。Linux侧已核对实际原始token存在、尺寸与xprof结果一致、目标实例/dispatch/PMC正确；本机没有运行Windows XCompute GUI，因此尚未验证GUI的指令匹配或逐指令Latency，不声称已完成动态瓶颈归因。

两个方案先各做3次CK与BLAS全输出比较，均8192元素/次、failed=0、max_abs=0，CK/BLAS redzone均通过。二进制SHA与11月6日的trial06b完全一致。新旧选核、源代码和历史性能表未变。

首次自动选CU时，旧方案仅800字节SQTT；原记录保留，不作为交付的指令分析对照。随后显式固定同一个CU重采双方，得到旧65920字节、新84832字节SQTT。这是采样覆盖修正，没有按性能快慢挑选结果。

采集命令含 `--enable-sqtt --enable-raw --target-se 0 --target-cu 0 --dispatches 18`，并保留 `--sections` 的原8项。每条GPU命令前均通过hy-smi空闲和内核服务门禁，统一锁串行；未发现MKFD/服务事件。没有重新计时、全量测试或恢复已暂停P6g。

特别注意：xprof replay显著改变benchmark事件计时；run.log中的PERF不是正式性能数据，不能据此更新R或分桶。

目录中脚本和command/run.log记录可用于核对命令；`.out`是额外导出的原始16位十六进制token，不是已解码的指令耗时。主下载包包含两份最终.perf、原始SQTT、results.txt、命令、README和summary。
