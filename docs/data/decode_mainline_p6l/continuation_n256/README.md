# N256 双槽／三槽：完整代表发现与 SQTT

本阶段为独立 H3 流水的有限 N 面板／预取深度研究，全部经 GPU4 串行、同二进制和同 epoch 完成。
正式 benchmark SHA256：`0d078209b33c37cc6eaef1ff16d0132615227e6cad4733cc121bd8b16ca9d876`。38 资格作业、55 个三轮配对、438 次全输出比较及真实 guard 全部通过。

- d2：19 条代表，耗时几何平均候选／当前 CK 1.042323；改善超过 2% 共 7 条，回退超过 2% 共 11 条，高 CV 0 条。
- d3：19 条代表，耗时几何平均候选／当前 CK 1.034429；改善超过 2% 共 4 条，回退超过 2% 共 8 条，高 CV 1 条。

完整范围存在回退，当前不采用；这些三轮数据不是七 block 验收。没有根据个体赢家更新原 P6k、BLAS 或 Excel，目标计数仍为 0/50、0/30。原表 BLAS 来自冻结 GPU0 Native GroupGEMM，不能称为同卡新鲜 BLAS；所有新 CK 对照来自 GPU4 同二进制。

SQTT 共 12 份，下载收据为 profile_delivery_batch11.json；XCompute 打开说明和原始相对路径另附。深度控制 base 为新 D2 实验核，不能称为当前最佳 CK。GUI 未在本机运行；perf 时长不作正式成绩，wave 累计时钟不能相加充当墙钟。

n256_profile_rule_metadata_erratum.json 说明预登记 rule 文件中沿用的 K64 描述字段。实际已冻结 runner 明确强制完整 55 对、19 代表的 D2/D3 GM 选择及 805 深度控制；未在测量期间改变规则、捕获选择或验收方法。

paired_results_with_tflops.csv 保留全部 55 对、配置／实际实例、TFLOPS、三轮各摘要与高 CV；raw_jobs.tar.gz 按成员 SHA 保存全部日志、runner 精确版本和 manifest。大型 perf、实际编译源码中用户 overlay、二进制与厂商 ISA 不入 Git；现场 compile/build11/compiled_sources 保留实际 25 份编译输入。

原会话期限保持；本次新增窗口的新实验截止北京时间 20:10:44、硬截止 22:10:44。未跑 1193/881 全量 GPU，没有 push；自有实验源码单独提交，原用户 tracked/untracked 与 readfirstlane 保留。
