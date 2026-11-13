# 低 LDS N128 双槽与属性分派：完整代表发现及 SQTT

本阶段为独立 H3 流水的低 LDS N128 双槽及属性规则研究，全部经 GPU4 串行、同二进制和同 epoch 完成。
正式 benchmark SHA256：`a8180db82315d1cb6b875ee8b9e1ef11933fa00e81fbf835636e848da3e947ab`。42 资格作业、59 个三轮配对、474 次全输出比较及真实 guard 全部通过。

- pure：23 条代表，耗时几何平均候选／当前 CK 1.173542；改善超过 2% 共 1 条，回退超过 2% 共 18 条，高 CV 0 条。
- auto：23 条代表，耗时几何平均候选／当前 CK 1.089295；改善超过 2% 共 8 条，回退超过 2% 共 10 条，高 CV 2 条。

本档仅是完整三轮发现，尚未采用；范围内的全部收益／回退保留，后续候选仍须完整属性域七 block 验收。没有根据个体赢家更新原 P6k、BLAS 或 Excel，目标计数仍为 0/50、0/30。原表 BLAS 来自冻结 GPU0 Native GroupGEMM，不能称为同卡新鲜 BLAS；所有新 CK 对照来自 GPU4 同二进制。

SQTT 共 14 份，下载收据为 profile_delivery_batch13.json；XCompute 打开说明和原始相对路径另附。几何控制 base 为原 N256 D2 的实验核，不能称为当前最佳 CK。GUI 未在本机运行；perf 时长不作正式成绩，wave 累计时钟不能相加充当墙钟。

本轮在首个GPU前固定两个policy各23代表（19旧代表+按最低CTA规则选择的4条原shape）、固定结构覆盖与缺失类别，以及805/1009新旧几何对照。Ntile、wave分工和LDS足迹耦合改变，不称为纯N因果。

paired_results_with_tflops.csv 保留全部 59 对、配置／实际实例、TFLOPS、三轮各摘要与高 CV；raw_jobs.tar.gz 按成员 SHA 保存全部日志、runner 精确版本和 manifest。大型 perf、实际编译源码中用户 overlay、二进制与厂商 ISA 不入 Git；现场 compile/build13/compiled_sources 保留实际 25 份编译输入。

原会话期限保持；本次新增窗口的新实验截止北京时间 20:10:44、硬截止 22:10:44。未跑 1193/881 全量 GPU，没有 push；自有实验源码单独提交，原用户 tracked/untracked 与 readfirstlane 保留。
