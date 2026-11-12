# H2共享A/私有B：局部收益，未整域采用

物理GPU4、统一锁/MKFD门禁、epoch0；同一正式benchmark/default selector，baseline沿P6k每条当前可启用配置同binary重新计时。38资格、39组完整三轮AB/BA对照、12份成对SQTT均有效，无MKFD/恢复；不是1193全量或七block确认。每臂先全输出3次及真实guard，再warmup10/samples30/单launch/include_args_build0，三轮汇总轮中位数的中位数。

K16D8：8条代表6条改善>2%、2条回退>2%、5条点估计改善≥5%；candidate/base几何均值0.972905，逐case耗时和比0.981352。ID620/621/805/948/1009/1072变化约−8.210/−5.966/−6.470/−2.408/−7.452/−7.281%；ID633/788约+12.447/+5.593%，高CV照实保留。K16D4/K32D4几何均值1.149296/1.152016；短线9条为3改善、4回退、2在±2%，GM1.019538。6个域外控制变化均在±2%。完整域不能据此采用，当前50/30稳定/归因目标未验收。

Profiler：ID621/805候选VMEM每wave196/388，wait/wavecycle .096/.104，对应H1原型644/1284和.770/.819；该架构改变了指令/等待行为；同时改变加载、缓存策略与同步安排，不能把全部收益单独归因于共享A。剩余LDSbankconflict/active约.667（M64）、.501（M2）、K32 .700，TCP pending和translation misses仍高。HBM在长K双方均约一份输入，不可称B重复流量就是根因。下一步仅按证据做XOR_A与提前B两个因素的受控H2修正，设计另见现场tuning_hypothesis.json；不混入本阶段成绩。

第三批下载包及SHA见profile_delivery_batch3.json，含固定代表及按事先规则补齐的gain/flat/loss共六组SQTT、原始名字/依赖、kernel/dispatch和Source PC/Wavefronts说明。未在本机运行Windows XCompute；profiler时长、累计wave时钟均不是正式成绩。perf/二进制/厂商ISA未提交Git。

完整原始作业与门禁见raw_jobs.tar.gz/jobs.json，所有39对/正负结果、高CV保留。build3原1574+H1新增30 shader机器码保持，9个H2 gfx936核无scratch/spill。源码对应本地实验提交b8f4e7f及保留的用户overlay，source_sha256记录实际编译字节；工作区新修正不反向覆盖本轮数据。当前原1193表、冻结BLAS和旧Excel均不修改。
