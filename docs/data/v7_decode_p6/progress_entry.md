### P6a（2026-11-03）decode 四形状 profiler 诊断

用户授权范围：先做交接 §4.1 的 ID76、ID681、ID633 和新对照 ID358，形成根因结论后停下。**本阶段完成，未进入方案实现。**

- 生产代码基线为 `8b05927`；没有改 kernel、selector、epilogue、实例、接口或构建清单，没有重编、没有跑 1178 全量、没有 push。readfirstlane 保留未提交，`git diff | md5sum = 4c7f49a5c7592767d53195a713db6b8a`，benchmark/xprof/BLAS bridge 二进制 SHA256 前后不变。
- 完整报告及命令、表格、测量边界：`/workspace/v7_coverage/p6/report.md`；仓库中的小型证据副本：`docs/data/v7_decode_p6/README.md`。原始日志和 .perf 在 `p6/runs/`，不放入 Git。
- 命令：`ulimit -c 0` 后执行 `HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6/run.py timing`，然后同环境 `run.py profile`；CPU 分析为 `python3 /workspace/v7_coverage/p6/analyze.py` 和 `write_report.py`。
- 物理卡 4 串行，58 次门禁全部 HCU%=0。有效记录 56 条：24 条三轮 CK/BLAS 交替计时、24 条三轮 GC、8 条一轮 TLB/UMC。计时两边都为单次 launch，warmup=10、samples=30。BLAS 固定重放冻结算法，不重新调优。`CAMPAIGN_CHECK_PASS valid_jobs=56 timing=24 gc=24 tlb=8 padded_mmop=24/24`；另对全部 48 个 GC dispatch 核验 MMOP 和 wave 数通过。

**已验证的结构事实**

| ID | CK/BLAS CTA | CK/BLAS waves | CK/BLAS MMOP 指令 | CK/BLAS CU busy（三轮中位数） |
|---|---|---|---|---|
| 76 | 32/128 | 256/512 | 393216/98304 | 0.297/0.492 |
| 681 | 24/48 | 192/192 | 786432/196608 | 0.226/0.223 |
| 633 | 32/128 | 256/512 | 1048576/262144 | 0.281/0.469 |
| 358 | 768/1536 | 6144/6144 | 25165824/12582912 | 0.916/0.928 |

- BLAS 桥接设 m=N、n=M，tile 必须换回逻辑 M/N。ID681 的 MT128x64 对应逻辑 64×128，实际 48 CTA。
- 三条问题 shape 同时有低并行度和更大的每 CTA 工作量：CK 只有 24/32 CTA，80 CU 上活跃 CU 上限为 30%/40%；MMOP 总数为 BLAS 的 4 倍，而实际 HBM 读量接近。CK/BLAS 平均 wave 生命周期比约 1.18×、2.05×、1.73×，对照 ID358 为 0.98×。M tile 除了影响 CTA 数，还影响补零行的实际计算量，应纳入新变体评估。
- 短 K 固定项：三条问题 shape 均有 `(VALU-MMOP)/wave=709.75`，占 VALU 计数在 K768 为 31.6%、K2048 为 14.8%；`VMEM/wave=2*(K/16)+10.25`。源码是 4 级 ring 加 1 个预装 stage，K loop 尾端仍把无效预取改读 stage0。额外 5 stage 加载相对必需 stage 数为 10.4%/3.9%。**这些是指令/请求占比，不是耗时占比；浅流水收益未验证。**
- 三条问题 shape 的 CK DRAM credit stall 全为 0，全部 CK UTCL2 miss=0；对照 ID358 credit stall 约 0.13。未观察到 P4a 大 K 的 HBM 读放大/拥塞模式。

**计时与边界**
- 三轮同卡单次 launch 的 CK/BLAS ms：ID76 0.062162/0.024320，ID681 0.071121/0.034000，ID633 0.066241/0.041600，ID358 0.664879/0.699679。小 kernel 波动明显，范围/CV 全部保留，不用最快轮替代中位数；冻结基线仍为卡 0、20 次 launch 的另一口径。
- 小 shape 的 GUI/1.6GHz 与 AQL 时长不一致，多 pass 的派生比值只作诊断，不作精确的损失时间分解。稳定的执行量是主要结构证据。
- BLAS 12/12 次全输出参考比对 `failed=0 max_abs=0 guards_ok=true`；CK 12/12 次 CPU 抽样和 guard 通过。没有生产代码改动，本阶段未重跑 selector/device_args/fused 回归。
- 两条准备阶段失败记录保留：CK 输出字段解析（benchmark 实际 PASS）、BLAS 预加载库冲突（bridge 尚未执行）；修复诊断脚本后断点续跑，不计入 56 条有效记录。

**后续建议（尚未实施）**
- 并行评估 N 切分与短 K 固定开销，并核对 M tile 对实际 MMOP/LDS/寄存器工作量的影响。新的小 tile 仍须按 V7 重建与 A/B 验证，旧 4002 更慢的结论保持。
- 104/45 条是达到 320 CTA 阈值的覆盖率，320 是经验分段。三条问题 shape 仅切 N=64 可达到 128/96/128 CTA，虽不足 320，仍跨过 80 CU；未达 320 的 45 条仍可能从 N 切分获益，达到阈值也不保证性能改善。
- 新实例无列谓词时明确 `N % Tn == 0` 及选核/回退契约。prefill 保持，N=1232 仍留待用户决定。
- 按用户要求在此停下，等待确认后才进入下一阶段。
