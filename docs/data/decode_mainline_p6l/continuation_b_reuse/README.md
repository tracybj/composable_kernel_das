# B复用 N256 双槽／三槽：完整代表发现与 SQTT

本阶段为独立 H3 流水的有限 N 面板／预取深度研究，全部经 GPU4 串行、同二进制和同 epoch 完成。
正式 benchmark SHA256：`6108cce9a7e056d0bcad78f122e92d61884ad27a3f225075f879f3dc19f850db`。38 资格作业、59 个三轮配对、462 次全输出比较及真实 guard 全部通过。

- d2：19 条代表，耗时几何平均候选／当前 CK 1.015139；改善超过 2% 共 9 条，回退超过 2% 共 7 条，高 CV 1 条。
- d3：19 条代表，耗时几何平均候选／当前 CK 1.034028；改善超过 2% 共 6 条，回退超过 2% 共 10 条，高 CV 3 条。

本档仅是完整三轮发现，尚未采用；范围内的全部收益／回退保留，后续候选仍须完整属性域七 block 验收。没有根据个体赢家更新原 P6k、BLAS 或 Excel，目标计数仍为 0/50、0/30。原表 BLAS 来自冻结 GPU0 Native GroupGEMM，不能称为同卡新鲜 BLAS；所有新 CK 对照来自 GPU4 同二进制。

SQTT 共 14 份，下载收据为 profile_delivery_batch12.json；XCompute 打开说明和原始相对路径另附。分工控制 base 为原 N256 的同深度实验核，不能称为当前最佳 CK。GUI 未在本机运行；perf 时长不作正式成绩，wave 累计时钟不能相加充当墙钟。

本轮 rule 在捕获前固定：两深度各 19 代表，固定四 shape 和缺失类别，另采 ID805 同深度原分工／B复用直接控制；预取深度相同，生产者不变。

paired_results_with_tflops.csv 保留全部 59 对、配置／实际实例、TFLOPS、三轮各摘要与高 CV；raw_jobs.tar.gz 按成员 SHA 保存全部日志、runner 精确版本和 manifest。大型 perf、实际编译源码中用户 overlay、二进制与厂商 ISA 不入 Git；现场 compile/build12/compiled_sources 保留实际 25 份编译输入。

原会话期限保持；本次新增窗口的新实验截止北京时间 20:10:44、硬截止 22:10:44。未跑 1193/881 全量 GPU，没有 push；自有实验源码单独提交，原用户 tracked/untracked 与 readfirstlane 保留。
