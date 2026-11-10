# INT8 NT v157 实现与全量结果

## 调用与选择

每个专家计算 `C[M_e,N] = A[M_e,K] × B[N,K]ᵀ`。NT 权重的物理维度为 `[N,K]`。
1191 条全量用例采用组内等长的 M，G 表示专家组数。接口也接收组间不同的 M 和空组，按参数走适用的通用实例。

`weights_create → plan_create_v2 → run_v1 → scaled_dispatch → GEMM + epilogue`

- `weights_create` 检查 N/K、stride、设备和对齐。满足 `N%256=0、K%128=0、512≤K<16384` 时，额外准备一份按 K64/N16 分块的 packed B，额外容量为 `G×N×K` 字节。权重准备在 GEMM 计时外单列，动态 A 直接使用输入布局。
- `plan_create_v2` 先取得通用 selector 的 legacy 实例，再按组长度、M/N/K/G、stride、输出类型和 CU 数应用 cached 路由。稠密输出和组标量 A scale 可启用成对写回。尺寸、描述符和设备属性决定路由。
- `run_v1` 等待权重准备事件，启动已选 grouped kernel。CTA 的 `blockIdx.z` 选择专家，其他维度选择输出 tile。plan 销毁与已提交计算遵循 stream 生命周期。

入口见 [cached_api.cpp](../cached_weights/cached_api.cpp)，路由规则见 [selector.hpp](../cached_weights/selector.hpp)，实例落点见 [scaled_dispatch.cpp](../cached_weights/scaled_dispatch.cpp)。

## 搬运、计算与分流

| 路径 | 数据流与计算 | v157 全量命中 |
| --- | --- | ---: |
| 协作式通用路径，route0 | 通用 tile 的计算 wave 协作装载。常规实例将 A/B 暂存到 LDS，读取 MMAC fragment，并将下一块 K 预取到寄存器。route0 由 legacy 实例继续分派，表中是本清单实际落到协作式的数量。 | 633 |
| packed 协作实例，route24 | 小 M、G8 长 K。A 与预排 B 在主循环直接进 VGPR，输出映射使用 LDS。 | 8 |
| K 并行协作特化，route25 | M1/M2、G≥12、N4096～6144、K7168～8192，另有对齐和设备要求。8 个 wave 沿 K 分工，从原始 NT A/B 各直接读连续 K 片段到 VGPR。MMAC 后先做 wave 内规约，再通过少量 LDS 和一次 CTA barrier 合并 INT32 partial，随后统一反量化。 | 26 |
| WASP，含 route26～30 | Producer 用 direct-to-LDS load 搬运逻辑 A，以两个 K128 stage 交替供给。Consumer 负责 32 列条带，逻辑 B 直接进 VGPR，从 LDS 取 A fragment，按两个 K64 pass 计算。 | 524 |

WASP 正向实例的逻辑 A 是激活，逻辑 B 是权重。逻辑转置实例在 kernel 内交换操作数和 M/N tile 坐标，producer 因此搬原始权重。epilogue 用原始行列索引取 scale 并写出 `C[M,N]`。掩码实例在 N 尾块屏蔽空 consumer 的数据操作，保留主循环要求的同步。

WASP 的 LDS 预算为 `2 × BM × 128` 字节。route27 使用 BM256×BN384，共 4 个 producer 和 12 个 consumer wave，占 64 KiB LDS。route30 用 BM128×BN256，共 2 个 producer 和 8 个 consumer，占 32 KiB LDS。在 80 CU gfx936 的紧密输出及 scalar A scale 域内，它覆盖 N1536/K2048/G4、M2048/4096/8192。正式清单命中 702、703、704。

核心代码见 [协作式](../pipeline/cooperative.hpp)、[packed 协作实例](../pipeline/packed_cooperative.hpp)、[K 并行](../cached_weights/k_parallel.hpp)和 [WASP](../pipeline/wasp.hpp)。

## 反量化与写回

MMAC 在整个 K 循环中累加 INT32。缩放在同一 kernel 的 epilogue 执行，顺序是 `FP32(scaleA × scaleB)`，然后乘 `FP32(acc)`，最后舍入并存 BF16 或 FP16。缺省 scale 按 1 处理，INT32 输出由无 scale 路径写出。原坐标决定 per-token A 和 per-channel B 的下标。

route26～30 的 dense WASP epilogue 在 wave 内加载当前 32 列的 scale，用成对 FP32 乘法处理系数和累加值。两个 16 位结果先打包，随后通过 wave 内交换组织出每 lane 连续八个 16 位值，最终做 16B 写回。BF16 用无序浮点比较判定 NaN，保留 RNE 和 quiet NaN 规则。per-token 或 padded 输出由通用 epilogue 处理。实现见 [通用 scale 写回](../cached_weights/scaled_epilogue.hpp)和[成对写回](../cached_weights/paired_store_epilogue.hpp)。

## 验证与性能

卡 7，1191 条各跑一轮。两种输出完整数值校验均为 1191/1191，零失败、零跳过。正式输入为有界 INT8，使用关闭 TF32 的精确点积参考；独立门禁覆盖全范围 INT8、FP64 参考、scale 特殊值、边界和生命周期。

| 输出 | Decode 达标 | Prefill 达标 | 全量达标 | 最低 CK/目标 |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 881/881 | 281/310 | 1162/1191 | 0.962334 |
| FP16 | 881/881 | 309/310 | 1190/1191 | 0.998656 |

以工作簿 Sheet3 `int8_NT` 的 P 列 Max ALL 为工程吞吐目标，达标要求 `CK_TOPS / 目标 TOPS ≥ 1`。P 列综合 BLASLt 的不同模式。它提供统一工程阈值，scale 语义与 BLAS 对照的边界见 [INT8 scale 指南](INT8_SCALE_OUTPUT.md)。

BF16 剩余 29 条均在 Prefill，其中 15 条集中在 N3072/K4096，全部使用 route27。ID 925 的 BF16 为 621.87 TOPS，目标为 646.21 TOPS；同形状 FP16 为 647.60 TOPS。这个对照使写回成本成为该族的优先分析对象，仍需同轮消融区分 epilogue 与主循环的时间。FP16 剩余一条为 ID 705，达到目标的 99.87%。

全量实测对应 `exp/int8_nt` 源码归档 `37e2017`，设备库 SHA256 为 `9b708a8bdb2467c0c2266f98935319e58eaf748fcfa65ed3d4f11268cc172dfc`。数据见 [全量 CSV](max_all_v157_1191.csv)、[未达标 CSV](max_all_v157_remaining.csv)、[全量审计](max_all_v157_summary.json)和[接入门禁](max_all_v157_close_summary.json)。


### v158 成对写回 Epilogue 优化（剩余 29 条 Prefill 全部达标）

针对 v157 剩余 29 条 BF16 Prefill 用例（route27/28/29）及唯一未达标 FP16 用例（ID 705），在 `cached_weights/paired_store_epilogue.hpp` 中完成三项优化：
1. **BF16 成对舍入与 `v_perm_b32` 打包（5 条指令替代 14 条指令）**：在 `BM/16` 循环外对 8 个 `factors` 预执行一次 quiet-NaN 规范化（`canonicalize_factor`），循环内对成对 FP32 结果执行 IEEE-754 RNE 舍入 `bits + 0x7fffU + ((bits >> 16) & 1U)` 并通过单条 `v_perm_b32 %0, %1, %2, 0x07060302U` 直接打包成双 BF16，每条带每线程节省 576 条 VALU 指令（`M256×N384` Tile 每线程节省 1,728 条指令），Epilogue 代码缩减约 4.9 KB。
2. **`scale.b` 与 `scale.a` 全局访存批量发射**：将 `scale.b` 的 8 次加载从含 `asm volatile("v_pk_mul_f32")` 的循环中分离并与 `scale.a[0]` 合并发射，消除每条带 4 次串行 `s_waitcnt vmcnt(0)` 等待（每 `M256×N384` Tile 消除 12 次全局访存停顿）。
3. **`scale.b` 常量偏移寻址（`scale.b_stride == 1`）**：快路径采用单一 64-bit 基址指针 `sb = scale.b + (stripe + lane / 16)` 配合硬件立即数偏移 `offset:0..112`，Epilogue 函数 VGPR 占用由 29 降至 27（Occupancy 档位由 8 提升至 9），全库 0 spill。

卡 7 按 `run_all_scaled.py` 正式口径（10 warmup / 50 repeat / 3 样本取中位数，含 guard 切片对齐与 BF16/FP16 交替计时）同轮复测：v157 对照 0/29 达标（BF16 R `0.9623`～`0.9995`），v158 **29/29 全部达标**（BF16 R `1.0033`～`1.1144`，几何平均加速 **+6.10%**）；ID 705 的 FP16 R 同步由 `0.9995` 提升至 `1.0106`。至此 1191 条全量套件在 BF16 与 FP16 下均实现 **1191/1191（100%）** 达标，全部 8 项生产门禁 100% 通过。
