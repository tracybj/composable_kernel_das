# BF16 NN Decode 独立主干研究：性能目标未达成

当前没有新属性域通过采用验收：稳定弱项改善 **0/50**，可归因跨桶 **0/30**。原 P6k 数据、冻结 BLAS、1193 个 shape 和原 Excel 保持不变；本轮新 GPU 门禁已关闭，Goal 仍未完成。

- [当前收尾报告：9 个续研阶段、348 个三轮配对](continuation_closeout/README.md)
- [逐项验收：50 条要求、证据及剩余缺口](acceptance_audit_20261113/README.md)
- [设计因素、资源、ISA 与 profiler 证据矩阵](continuation_closeout/readonly_reports/mechanism_matrix.md)
- [最新双累加候选逐 shape 数据，含 TFLOPS；候选未采用](continuation_accum/paired_results_with_tflops.csv)
- [各阶段实测分桶，全部未采用](continuation_closeout/fresh_buckets_by_stage_variant_NOT_ADOPTED.csv)
- [零替换的 P6k 历史投影；不是新鲜 1193 复测](continuation_closeout/projection_P6k_UNCHANGED_NOT_FRESH1193.csv)
- [原 24 小时窗口报告与 166 条七 block 否定证据](final_closeout/closeout/README.md)

旧窗口报告中的“最后调度原型仅编译”属于当时的状态；该候选已在新增窗口完成 GPU 资格、三轮对照和 SQTT。两个时点分别保留，不能把旧说明当作当前状态，也不能把后续阴性结果改写成采用。

原 P6k 的互斥桶为：prefill 312 条中 292/20/0/0/0/0，decode 881 条中 604/263/14/0/0/0，阈值依次为 R≥0.99、0.8≤R<0.99、0.5≤R<0.8、0.3≤R<0.5、0.2≤R<0.3、R<0.2。R 始终用冻结 Native GroupGEMM BLAS 耗时除以 CK 耗时。新的同卡 Native 诊断另列 R_live，保留 CPU 提交流程差异，不覆盖冻结 BLAS。

所有实测阶段分别保留二进制、epoch、完整配置、实际实例、精度与 guard、轮次、CV、原始日志和失败；不同构建的赢家不拼成“当前最好”。第 14 批 profiler 是部分交付，其应用中止和稀疏尝试均保留。其余下载包、具体 kernel/dispatch/CU 和 XCompute 查看位置见收尾报告及各包 README；本机未运行 Windows GUI。

本轮仅物理 GPU4 串行，本地实验提交，不 push、不合入。提交截止和硬截止保持原授权，不因自动继续而延长。后续需要新的有效研究窗口才能开展新的优化、编译或 GPU 验证；当前报告命令只读处理现有证据，不解锁旧队列。

统一跨阶段断点续跑入口尚未完成验收；源码复核发现重新冻结会改变 manifest 并使原记录跳过条件失效，具体见[只读源码证据](acceptance_audit_20261113/resume_source_audit.json)。当前已测数据不受影响，旧入口不应被当作已验证的通用续跑工具。
