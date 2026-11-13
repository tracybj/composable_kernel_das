# N128 四 wave 完整 M 所有权与 B 复用：完整代表发现及 SQTT

本阶段为独立 H3 流水的四 wave 完整 M 分工及 B 复用研究，全部经 GPU4 串行、同二进制和同 epoch 完成。
正式 benchmark SHA256：`5aba313b48c54cf64182a7855a91ad25498edd220c0198fe5d9d4b25da5e1151`。35 资格作业、27 个三轮配对、261 次全输出比较及真实 guard 全部通过。

- scope_wave4：16 条代表，耗时几何平均候选／当前 CK 1.183817；改善超过 2% 共 1 条，回退超过 2% 共 13 条，高 CV 1 条。

本档仅是完整三轮发现，尚未采用；范围内的全部收益／回退保留，后续候选仍须完整属性域七 block 验收。没有根据个体赢家更新原 P6k、BLAS 或 Excel，目标计数仍为 0/50、0/30。原表 BLAS 来自冻结 GPU0 Native GroupGEMM，不能称为同卡新鲜 BLAS；所有新 CK 对照来自 GPU4 同二进制。

SQTT 共 4 份，下载收据为 profile_delivery_batch14.json；XCompute 打开说明和原始相对路径另附。几何控制 base 为原 N128 D2 八 wave 实验核，不能称为当前最佳 CK。GUI 未在本机运行；perf 时长不作正式成绩，wave 累计时钟不能相加充当墙钟。

SQTT仅完成原8对中的2对（4份有效perf），其余未完成。ID686在CU0/1/2/3/7稀疏，CU15 profiler内应用rc134中止，原因未充分定位。没有放宽4096字节门槛；失败完整保留，独立新进程dmesg/hy-smi门禁健康，无已观察MKFD/注册事件。原停止状态保留，新原始性能数据未受修改，也不采用任何候选。

本轮在首个GPU前固定单一M17..64属性规则，采用先前23代表中全部16条在域shape、7个回退控制与4个同N128/K32/D2/CTA网格/LDS的W8/W4分工对照。wave数、M所有权/B复用和寄存器需求共同改变，不称为单指令因果。若发现阶段通过，仍须完整96原shape/53弱项及24未见、固定7block独立确认；不按赢家ID分派。

paired_results_with_tflops.csv 保留全部 27 对、配置／实际实例、TFLOPS、三轮各摘要与高 CV；raw_jobs.tar.gz 按成员 SHA 保存全部日志、runner 精确版本和 manifest。大型 perf、实际编译源码中用户 overlay、二进制与厂商 ISA 不入 Git；现场 compile/build14/compiled_sources 保留实际 25 份编译输入。

原会话期限保持；本次新增窗口的新实验截止北京时间 20:10:44、硬截止 22:10:44。未跑 1193/881 全量 GPU，没有 push；自有实验源码单独提交，原用户 tracked/untracked 与 readfirstlane 保留。
