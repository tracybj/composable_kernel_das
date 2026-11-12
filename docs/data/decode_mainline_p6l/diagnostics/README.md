# P6l：同卡诊断及首个独立主干准备

这是阶段证据，性能Goal尚未完成。新主干尚无GPU精度或性能成绩；正式benchmark仍在构建。这里只归档已验证的16条代表、4组CK/Native采集，以及CPU实现检查。

- 同卡诊断：物理GPU4，16条，当前最佳CK与固定算法Native各全输出及三轮交错计时；128作业全部有效、epoch0，无MKFD或恢复。144次全输出检查、405,577,728元素；四条有任一臂CV>5%，均保留。
- CK/Native每轮warmup10/samples30/repeat1。Native GroupGemm::run与CK直接launch的主机提交路径仍不同；新R_live只用于诊断。冻结GPU0/repeat20参考继续计算R_frozen，不回填历史BLAS。
- 第一批profiler：预声明ID76/621/805/1072，两臂各计数及精确末dispatch SQTT，16作业全部有效，8份SQTT均核对token尺寸、目标kernel/dispatch和真实资源，actual code反汇编unknown0。没有在本机打开Windows XCompute。

|代表|同轮CK ms|Native ms|已验证事实|仍需验证／对应设计|
|---:|---:|---:|---|---|
|76|0.021440|0.024000|CK64CTA、48KiB；Native128CTA、32KiB，两者有效MMAC比例均1/64；本轮CK反而更快|不能凭历史R或CTA数认定根因。短K测试较浅环及N64/N32|
|621|0.171683|0.132799|两者256CTA；CK有效MMAC比例1、Native0.5，CK仍慢；HBM约156.90MB对142.62MB|额外流量属于A还是B、操作数等待的贡献未证实；减少MMAC不是充分条件|
|805|0.513762|0.433360|CK256CTA/212VGPR/64KiB、有效MMAC0.25；Native512CTA/188 allocated VGPR/64KiB、有效MMAC1|验证全M输出块、降低LDS和寄存器资源，以及两层重叠|
|1072|0.818802|0.752400|CK有效MMAC2/256，Native2/64；HBM均接近一份输入|不能宣称当前V7存在大量重复B读取；新实现也必须保留B复用|

上述计数来自PMC replay，缓存行为会被采集影响。effective-MMAC比例表示有效FLOPs/执行MMAC对应FLOPs，不是墙钟时间占比；wait计数也不是单一因果证明。

首个结构H1：一个CTA覆盖完整M≤64，各N wave把B面板复用于所有M片段；A寄存器、B私有LDS，Global环与LDS操作数预读分开。D4/D8/串行操作数属于受限变体，短K另测D2的N64/N32。独立于V7框架，可在数据反证时淘汰；并不预定A寄存器一定优胜。

独立TU双架构编译23.07秒成功。gfx936的15个实例均无scratch/spill/CTA barrier；主线D4的静态VGPR为37/53/89，动态LDS16KiB。382项新边界、旧Depth8及原selector测试通过，原1193配置仅做CPU选核核对，结果保持；141个拟测选择核对通过。公开5个struct定义及既有87个显式常数对HEAD不变，用户readfirstlane及12个未触及原文件SHA保留。完整构建及GPU确认待完成。

perf和厂商ISA仅在现场保留，不提交Git。下载包：
`/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/delivery/P6l_batch1_CK_Native_SQTT.tar.gz`

大小173,174,075字节，SHA256 `2f9967942eac2b663397fb76080e4a82e457f9af3f7a26187b4fc28c66ad6788`。
原包不变，另附XCOMPUTE_FOCUS.md和capture_provenance.json补充定位及采集命令。用户反馈异步处理，不构成阶段审批。

所有原始命令、输出、门禁、kernel事件及runner快照见raw_diagnostic_records.tar.gz。会话T0为2026-11-12 02:53:23 UTC，20h新提交截止、24h硬截止不变。新开关默认关闭；原P6g/P6i保持暂停，不跑1193全量GPU，不push。
