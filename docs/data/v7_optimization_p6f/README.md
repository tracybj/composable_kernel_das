# P6f：尾块、缓存遍历、decode 收尾（实验用，不合入）

本轮推荐限定属性域的 tail256 和 NoSLC/M01=5 缓存分支。新的 K16/K32
稳态/收尾流水已实现并通过精度，但同几何对照没有稳定耗时收益；159 条
既有 K32 扩围也有明显退化，不整体采用。全部实验默认关闭，只做本地提交。

- [最终报告](final_results/report.md)：根因证据、三项状态、全部范围和环境限制。
- [逐 shape 对照 CSV](final_results/per_shape_results.csv)：288 组完整三轮对照，
  含有效开关、实例、CV、epoch、二进制 SHA、精度和日志来源。
- [推荐策略增量 CSV](final_results/policy_overlay_incremental_1193.csv)：新鲜尾块
  16 条和缓存 8 条叠加历史当前可选路线；其余来源标签保留。
- [历史数值包络 CSV](final_results/historical_envelope_incremental_1193.csv)：
  历史当前可选基线与本轮完整三轮系列的数值最小值，包含未推荐实验及噪声内差异。
- [新增 15 条](final_results/new15_results.json)、[全部退化](final_results/all_regressions.json)、
  [独立计时诊断](final_results/launch_scope_results.json)、
  [最终审计](final_results/final_audit.json)。

两份1193表均为带来源的增量拼接，没有运行1193/1178 GPU全量，也不代表
统一策略的新鲜全量成绩。R保持冻结Native/正式单launch CK；20次launch
仅诊断，不进入R或包络。各自R≥0.99为903/1193、943/1193。

推荐的两个新属性分支可通过以下开关显式启用：

```bash
export CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1
export CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL=2
export CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID=1
```

尾块契约为common非ragged/empty、gfx936 BF16NN、M≤128、N≥128且N%8=0、
N%128≠0、K64..4096%16、G≤1024，加地址/网格范围。缓存契约为common
非ragged/empty、gfx936 BF16NN、1024<M≤2048、N>4096且N%256=0、
K3072..4096%16、G≤1024，加地址/网格范围。其他输入按其余现有开关回退。
原V7无列谓词契约不变，v1 ABI/GemmTransKernelArg语义不变。

上述三个开关只启用两个新分支；1193增量表中历史行可能使用其他既有开关，
逐行复现应使用CSV里的完整enable_env及原始命令。正式入口仍为
`build/bin/tile_example_grouped_gemm_device_args_benchmark`，默认selector，
warmup10/samples30、每样本单launch、include-args-build=0。

现场 `/workspace/v7_coverage/p6f/` 保存二进制、实际profiler和vendor ISA；
仓库仅归档小型自有脚本、派生证据及压缩日志。`final_results/manifest.json`
列出文件字节SHA；完整raw记录、服务异常、门禁、恢复与源版本均保留。
ID688旧回退对照的性能前精度门禁遗漏及整组替代见
`measurement_exclusions.json`，不可用初始checkpoint覆盖最终排除。

CPU分析可以读取现场记录重算；GPU复现必须使用新的授权会话及其期限，
不能重置本轮session或沿用已过期runner。仍仅允许物理4/0，GPU4串行，
每命令独立门禁，异常立即停队列并按有界恢复流程执行。

历史checkpoint按时间保留：cache_checkpoint、cache_domain_checkpoint、
implementation_checkpoint、initial_results_checkpoint；最终排除和采用结论
以final_results为准。
