# K64双槽及wave分工：完整发现与负面结论

明确授权的12小时续研，GPU4串行、epoch0、binary d94d9808dede6ae6ba6e8cd332d9e6b91196795e08a96713b13278ccc6a59374。新实验截止北京时间20:10:44，硬截止22:10:44；旧会话期限保留。本阶段完成29资格、81次全输出和guard；46个完整三轮配对，原始日志审计通过。

四wave19代表GM候选/当前CK=1.081196，5条改善>2%、11条回退>2%、12条高CV；八wave14代表GM1.136593，1条改善、10条回退、3条高CV。集合不同，不能直接据两个GM排名。完整域及已登记低G范围都不采用；三轮发现和个体赢家不构成七block验收。稳定50/归因跨桶30计数均为0，没有新heldout或全1193/881测试。原P6k、BLAS与Excel不改。

两项K64配置减少了同步阶段，但寄存器和LDS代价明显。四wave621/805平均驻留约6.36/6.68→3.17/3.63。另预声明补采ID805的K32D4八wave6027/K64D2八wave6029直接控制：同M64/N128、48KiBLDS，实际驻留7.383→7.210，HBM557.6→562.8MB；wait比.170→.256、LDS冲突比.178→.380，SALU1303→680而VMEM194/LDS786/MMAC1024每wave相同，正式三轮慢15.3%。这支持继续区分等待、LDS访问和资源代价，计数本身不证明单一根因。K64四wave与旧6022的控制还耦合MMAC顺序，不称为纯K因果。

第10批4对8份SQTT，补批10b为上述实验K32/K64的1对2份；不是把6027称为当前最佳CK。两包原名/sidecar与XCompute说明保留，下载路径/大小/SHA在两份delivery收据，GUI未本地打开。perf耗时不作正式性能成绩，不把不同wave累计时钟相加。

本阶段407个作业（含补采2）全部终态，357次CK全输出失败0，实际元素数见audit_summary。raw_jobs.tar.gz含原始job记录、原字节过滤的日志、精确runner版本和manifest，每成员SHA核对；大型perf、二进制与厂商ISA不入Git。24份实际编译输入含用户overlay，仅保存SHA并引用现场compile/build10/compiled_sources，不代提交用户修改。自有9源码提交4eb728782b9c9a739a075cf08d157428af224c85（实验用，不合入），用户9tracked/10untracked/readfirstlane保留。

CPU归档helper一次基线不匹配和一次收据git status目录折叠均已保留并独立修复；未影响GPU、原始性能或用户文件。不将这些审计问题归为MKFD。新的N256后续实验单独冻结/计时，不混入本档。

逐对表paired_results_with_tflops.csv含46对全部ms/TFLOPS、原冻结Native GroupGEMM BLAS/GPU0来源、三轮median/min/max/CV和质量标记；不是每shape择优，也不把both精度参考当Native性能。K64D2与K32D4“相同缓冲跨度”指K轴128个元素，不是128KiB。
