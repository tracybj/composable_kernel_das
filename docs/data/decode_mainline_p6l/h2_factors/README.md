# H2布局与提前B因素：局部收益，未整域采用

全部74对完成；GPU4、统一锁/MKFD门禁、epoch0。控制与候选来自同一build4正式benchmark/default selector，保留P6k每shape当前可启用配置；每臂先全输出三次及guard，再三轮AB/BA，warmup10/samples30/单launch/include_args_build0。全部正负结果和高CV保留，未做七block稳定性确认。

固定主线K16D8、短线K16D4，对旧控制、XOR_A、EARLY_B及组合做受控比较。主线XOR/组合GM candidate/base为0.964332/0.964208，仍有回退；EARLY单独1.024329。短线各组合GM均>1，不采用、不计入50/30。完整数据见tuning_discovery_analysis.csv/json，不能拼接逐shape最佳值。

Profiler显示XOR+提前B的LDSbankconflict/active在ID621/805/1072约.414/.407/.254，较原H2下降，但不能单独证明整体因果。短76的profile kernel反向更快而正式三轮更慢，保留差别，不能用profiler代替正式计时。下一轮将验证新的协作B/三组操作数结构及同GPU代码的host入口对照；本归档不含其成绩。

最初DISABLE身份探针G3/27CTA只采384字节，实际kernel符合但按4096门槛停止，未做性能；旧75条完成资格记录及原始作业保留。仅三条身份探针G改32后重新冻结并完整重跑80资格，再74对；未修改C++/二进制/门槛，非MKFD。三个有效DISABLE身份SQTT各>700KiB，旧稀疏perf仍在现场，不用于指令分析。

第四批下载包见profile_delivery_batch4.json，含10份成对SQTT及3份有效DISABLE身份trace、XCompute的kernel/dispatch/Source位置和全部依赖。未在本机运行Windows GUI。原1574+H1/H2新增48 shader机器码全部保留，27个新gfx936 tuned kernel无scratch/spill。源码本地实验提交a383912；初始用户overlay保持，原始1193/冻结BLAS/旧Excel不改，无1193全量复测。

raw_jobs.tar.gz包含758个原始作业（含稀疏捕获）、门禁/进程/命令日志、相应冻结manifest与脚本版本；raw_manifest.json逐成员核对SHA。perf、二进制及厂商ISA不入Git。compiled_sources保留本批两份完全自有源码的实际编译字节；source_sha256记录其余实际构建输入，initial_tracked.patch.gz无损保留用户overlay。
