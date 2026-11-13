# P6l 新增 12 小时窗口：研究进展与证据索引

目前没有新的属性域通过采用验收，性能目标仍未完成。 原 P6k、冻结 BLAS、1193 shape 与原 Excel 保持不变；新的历史投影与新鲜实测分开保存。

本窗口 T0 为 2026-11-13 02:10:44 UTC（北京时间 10:10:44）；新实验截止 12:10:44 UTC，硬截止 14:10:44 UTC。全部等待、编译和恢复均计时；旧会话期限保持。

| 独立实验 | 变体 | 代表数 | 耗时 GM 候选/对照 | 改善 >2% | 回退 >2% | 高 CV |
|---|---|---:|---:|---:|---:|---:|
| phase | compact | 19 | 1.015421 | 8 | 4 | 2 |
| phase | wide | 14 | 1.031505 | 6 | 8 | 0 |
| k64 | compact | 19 | 1.081196 | 5 | 11 | 12 |
| k64 | wide | 14 | 1.136593 | 1 | 10 | 3 |
| n256 | d2 | 19 | 1.042323 | 7 | 11 | 0 |
| n256 | d3 | 19 | 1.034429 | 4 | 8 | 1 |
| b_reuse | d2 | 19 | 1.015139 | 9 | 7 | 1 |
| b_reuse | d3 | 19 | 1.034028 | 6 | 10 | 3 |
| low_lds | auto | 23 | 1.089295 | 8 | 10 | 2 |
| low_lds | pure | 23 | 1.173542 | 1 | 18 | 0 |
| wave4 | scope_wave4 | 16 | 1.183817 | 1 | 13 | 1 |
| noslc | noslc | 16 | 1.124657 | 2 | 12 | 2 |
| guard | noslc | 3 | 1.171838 | 1 | 2 | 1 |
| accum | accum | 16 | 1.123799 | 2 | 11 | 2 |

每行是一套冻结二进制内的三轮配对发现数据，GM 小于 1 表示候选更快。不同构建的赢家不拼成最终成绩；14、19、23 条代表集也不能混称同一集合。点估计改善不等于七 block 的稳定改善或可归因跨桶。

几何控制和深度控制保留在各阶段逐 shape CSV，不能将独立实验核对照冒充当前最佳 CK。原 BLAS 为 GPU0 冻结 Native GroupGEMM，本窗口正式 CK 为 GPU4 同卡两臂；早期新同卡 Native 诊断另列 R_live，不覆盖 R_frozen。

已验证的局限：先计算调度没有形成整体收益；K64 减少循环和 SALU 后仍有资源／等待代价；N256 的部分局部收益不能通过整域回退门槛；B 复用减少寄存器与 LDS 指令后，ID805 对照的实测平均驻留基本不变；低 LDS 提高驻留后，指令／事务代价仍可能增加；四 wave 与缓存位修订均保留全部回退。这些证据否定相应充分性假设，不能据单一计数宣称完整根因。

短 K／低 G 的两项原型、初始同卡 Native 诊断、先前 166 条七 block 否定证据，见旧会话 final_closeout；本窗口没有恢复其未完成队列。旧确认个体 28 条稳定改善／14 条归因跨桶不构成采用，不能填入本轮 50／30 计数。

最终采用仍须：冻结属性域的全部原 shape（含已有赢家）、至少 24 条未参与设计的合成边界、固定七个配对 block、全输出精度与真实 guard、选核及回退，以及无确定 >2% 回退。原 seed 6411926 已用；下一 seed 6411927 仅在可确认的新候选实际冻结时消耗。

以下 perf 包保留相对路径、代码与 SQTT sidecar。解压后先读各包 XCOMPUTE_README.md，再按 profile_manifest 打开目标 perf、dispatch 18 与对应 XCD/CU。Source 查主循环等待／DS read／MMAC，Wavefronts 对齐完整主循环；不同 wave 累计 Latency 不相加当墙钟。GUI 未在本机运行。

- 第 9 批：[P6l_batch9_phase_first_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch9_phase_first_SQTT.tar.gz)，10 份有效捕获，SHA256 `3314320967ae3e237f456f69f373c0a527ae577c25d7f1e274c39b638cf6f56c`。
- 第 10 批：[P6l_batch10_k64_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch10_k64_SQTT.tar.gz)，8 份有效捕获，SHA256 `733e699ff216ca4417a60da11356b9233ba1852a9026f8236b18091de00fe0bd`。
- 第 10b 批：[P6l_batch10b_k64_wide_depth_control_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch10b_k64_wide_depth_control_SQTT.tar.gz)，2 份有效捕获，SHA256 `9457a8885f96af87532e796f845fe4e6dee5bd002a01a34f41840d1366668bb7`。
- 第 11 批：[P6l_batch11_n256_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch11_n256_SQTT.tar.gz)，12 份有效捕获，SHA256 `dacf3c005fccff5b4c617cdc9649ea8c12f172711bb699e06eb1dc9a01a0831d`。
- 第 12 批：[P6l_batch12_b_reuse_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch12_b_reuse_SQTT.tar.gz)，14 份有效捕获，SHA256 `e49a20a859f54393ed86d68792aa56d75446062ab4bb26c632cea4678c71cd20`。
- 第 13 批：[P6l_batch13_low_lds_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch13_low_lds_SQTT.tar.gz)，14 份有效捕获，SHA256 `3be1c929cf353c347985a7b84ec9bff71b158b045e0f3a0cfb98df2a339ec0b7`。
- 第 14 批：[P6l_batch14_wave4_PARTIAL_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch14_wave4_PARTIAL_SQTT.tar.gz)，4 份有效捕获，**部分交付，原计划8对仅完成2对；失败保留。** SHA256 `a5f2d6532b2c223bf423b8267bea1ca7e98879d6dab2817bd1899ae6e637474d`。
- 第 15 批：[P6l_batch15_noslc_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch15_noslc_SQTT.tar.gz)，8 份有效捕获，SHA256 `6d71c58f480c3e8b97d039a83490d42bb68ec7798bcdd2abaf0ecfd5bce54156`。
- 第 16 批：[P6l_batch16_accum_SQTT.tar.gz](/workspace/v7_coverage/p6l_decode_mainline/continuation_20261113T021044Z/delivery/P6l_batch16_accum_SQTT.tar.gz)，8 份有效捕获，SHA256 `c8b9a82d829ad17bd3bf6f0604244d09194ae799d1296f3c5df9e8cfd481a5ea`。

本汇总为只读报告；实际状态、逐 shape TFLOPS、轮次、CV、实例、配置、epoch、源／二进制 SHA、原始日志与失败记录均保存在各阶段归档。没有 1193/881 全量 GPU，没有 push。
