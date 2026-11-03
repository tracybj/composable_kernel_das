# P6b：小 M V7 原型实验

范围：1178 集合中的 decode（M≤256）；先验证 64×128×16、64×64×16 四级流水。新增 5003/5004，实验开关 CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M=128/64；默认关闭。通过公开 device_args benchmark 和默认 selector 选核，不强制 expected_instance_id。

保持原 V7、5002 prefill 和 N=1232 路径。原 pipeline 文件不改，readfirstlane diff md5 保持 4c7f49a5c7592767d53195a713db6b8a。新实例只接受 gfx936 BF16 NN、common M≤256、common K∈[64,4096] 且 K%16=0、N%Tn=0、G≤1024、无 ragged/empty；拒绝时走已有回退。v1 接口、GPU 参数数组、workspace 和 stream 语义均保持。

## 数据流
- 256 线程 / 4 waves / 2×2 wave grid；wave tile 为 32×64 或 32×32，累计器分别 32/16 个 fp32 每 lane。
- 沿用 V7 的 MMAC、16-wide k-stage、3 组操作数 bank、4 级 LDS ring 和 12 phase 调度；按读取指令数调整 lgkmcnt 阈值，并在每阶段 block barrier 前排空 LDS 读取，保证下一步消费操作数和复用 ring 的时序。
- 物理 A panel=128×16、B panel=16×128。每线程仍加载 16 B，逻辑 M/N 以外的部分通过无效 INDEX 填 0，避免跨 tile/矩阵读取。LDS 为 4×8 KB=32 KB。
- B 改为每 wave 加载 4 个 K 行；LDS 行距为 256 B，m32x16 anchors 按该行距重算。A 保留原来半行交换与 b64 读取方式。
- 独立 epilogue 保持 BF16 RNE 和 row mask。N=64 的每 wave 写回行只有 4 个 16 B 单元，row XOR 掩码改为 3，避免原来的 XOR 7 造成越界/重叠。CPU 证明单 wave slab 映射双射，64×64/64×128 输出恰好覆盖 4096/8192 个元素。无外部 N 尾块支持。
- G≤16 仍走原标量 group scan，G>16 common M 仍走 O(1)；保持映射方式以便比较 tile 改动。

## 验证
先最小全输出 BLAS 比对，再做 M 尾块、K64/K80、G1024 和 N1232/ragged/empty/large-K/prefill 回退。性能用 base→128、base→64，三轮交替，逐条保存并支持断点。另测关闭实验开关的新旧二进制，并核对原 V7 kernel 字节。原型的三种路径在代表 shape 上采计数器，确认 MMOP/CTA 与设计一致。完整全量 1178 不在本阶段运行。

短 K 浅流水作为独立后续实验，本阶段保持四级，先隔离 tile 和资源变化。通过实验后更新 PROGRESS、本地提交（实验用，不合入），停下汇报，等待确认。

## 同步诊断（已执行）

初版 N=64 在 ID76 上出现偶发误差（重复3次分别 1/8/5 个 CPU 抽样点失败），N=64 K64 G4 同样可复现；N=64 K768 G1、N=128 K64 G8 对照通过。仅在最后两步排空 LDS 读取仍失败，记录于 tail_wait/。因此不进入性能采集。

在每阶段 VMEM 部分等待后、block barrier 前补充 lgkmcnt(0)，32KB 与临时64KB LDS配置分别对 K768 G4、K64 G4 连续20次全输出比对全部通过。该证据支持阶段间 LDS 完成/复用时序不足；没有逐指令追踪，不能声称已区分 operand bank 就绪和跨wave覆盖的各自贡献。最终移除尾部冗余等待和临时64KB host开关，保留每阶段排空，使用32KB；最终 stage_wait/ 的54用例、162次全输出比对 failed=0 max_abs=0。最后只改注释的重编前后 benchmark SHA256 一致：72e701492a3419ce552aeb0376e2c6c95151a375080a76939507fc8d1a38f548。
