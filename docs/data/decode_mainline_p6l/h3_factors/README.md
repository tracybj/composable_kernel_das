# K32预取深度与host策略解析：完整发现，尚未采用

全部116对完成，均由同一build7正式benchmark/default selector、GPU4/epoch1测得。每臂先全输出三次与真实guard，再三轮AB/BA，warmup10/samples30/单launch/include_args_build0；全部回退和高CV保留。112资格验证通过；本归档不包含七block独立确认，不计为50/30性能目标达成。

协作A/B K32的D4full/D4fast/D5full/D5fast八代表几何平均候选/对照为.945576/.927456/.940887/.933936，均6改善>2%、2回退>2%、1高CV。提前登记的M>=17或K>=6144子域中六代表D4fast/D5fast为.893778/.891733，平均差别很小，不足证明五级普遍更好。四个深度直接控制分别+2.949%、-2.699%、-1.277%、-1.176%，无高CV；必须按完整属性域决定采用，不能逐ID挑赢家。

短线XORfast/BOTHfast九代表GM为.969254/.963239，均无>2%中位回退，但5/6条高CV；未完成稳定性证明。四组同GPU kernel的host控制实际符号/机器码一致，正式变化为-1.555%/-0.786%/-10.813%/-5.846%，两条短线高CV。它们支持进一步核对host成本，不能把所有变化归因于流水本体。

28份有效SQTT组成14对，含固定主线/短线、gain/flat/loss样本、四组host和四组深度控制。29次采集有一份稀疏trace，保留并按预声明CU索引顺序重采；不改变shape、kernel或4096token门槛。下载包见profile_delivery_batch6.json，10188506字节/SHA429cf850856d86aab0263a3041b102b1d2c9357be6d57b428bf8903f01a2e247，GUI未在本机打开。ID788回退样本HBM接近一份输入，baseline8wave/64KiB与候选4wave/45KiB的驻留/请求并行度值得后续检查，不能仅从低wait比例推断更快。

build7用时3303.075秒；原1574+前序120 shader机器码不变，新增总数126。12个gfx936协作核无scratch/spill，三bank静态数据流核对通过。1750新选核、1063 Depth8及旧selector通过；51317 CPU策略场景及额外3456个有效几何/公开禁用位场景通过。源码自有增量本地实验提交5c9d3f7，初始用户overlay与readfirstlane保留，公开v1 struct/GemmTransKernelArg不改。

jobs.json.gz为完整1069作业ledger的无损gzip；raw_jobs.tar.gz保留原始作业/门禁/脚本版本/manifest和CPU日志，raw_manifest.json逐成员SHA核对。compiled_sources与build7实际15份输入逐字匹配，包含已记录的初始用户overlay。perf、二进制和厂商ISA不入Git；51k CPU测试输入由保留的确定性生成脚本和冻结P6k重建，不将大体积展开输入提交Git。

原P6k、1193形状表、冻结BLAS和旧Excel未改，无全量GPU复测。后续最终候选必须冻结可解释属性范围，包含全部原表成员及已有赢家、至少24个未见合成shape，并按预先固定七block规则确认。混合历史投影不能冒充新鲜1193结果。
