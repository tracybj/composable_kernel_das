# INT8 NT 两套核心流水整理与回归（v121）

## 范围与版本

整理基于 v120 结束时的源码 `81559d413ed2ef5698ae6ca0cd64f018b5941396`。
其中正式性能实现仍是 v112，v113–v120 的 M192、rotated LDS、stagger 等实验
没有提升剩余八条，未进入默认路径。本次是生产代码归一，不改变数学口径、
默认 selector 或已冻结 BLAS 目标，也不把 v120 实验 kernel 当成正式实现。

口径：gfx936，INT8×INT8→INT32，NT，无 scale；静态专家权重一次预排后复用；
动态 A 不预排；FP16/BF16 ABI 和分流保持原状。

## 两套核心流水

| 核心 | 实现位置 | 数据流与同步 |
| --- | --- | --- |
| Cooperative | `grouped_int8_nt_pipeline.hpp`；入口说明 `pipeline/cooperative.hpp` | 计算 wave 协作搬运 A/B 到 LDS，下一 K tile 预取到 VGPR；读完当前 LDS 后再覆盖、发布下一 tile。 |
| WASP | `pipeline/wasp.hpp` | 专用 producer 将 A 直接搬至两个交替 LDS 槽；consumer 直接读取 B 到 VGPR，用分批 A fragment 做 MMAC；每 K stage 使用统一的发布/退役协议。 |

五份 WASP 正式主循环已合为一份，原 m32/m64/m128/wide_n 和 cached header
仅保留别名。原 raw API 的 WASP 同样复用该核心。历史 split-consumer 实验在
实验分支原 revision 上复现，不再作为生产 header 中的独立流水维护。

| 实例族 | BM×BN×BK | P/C waves | A LDS | A-read batch |
| --- | --- | --- | --- | --- |
| packed M32 | 32×64/256×128 | 1 / 2或8 | 8 KiB | 1或2 |
| packed M64 | 64×64/256×128 | 1 / 2或8 | 16 KiB | 2 |
| packed M128 | 128×128/256×128 | 2 / 4或8 | 32 KiB | 4或2 |
| raw/transpose/packed M256 | 256×256×128 | 4 / 8 | 64 KiB | 4 |
| packed wide N | 256×384×128 | 4 / 12 | 64 KiB | 2 |

`cached_weights/core_problem.hpp` 描述统一 Problem，`instances.hpp` 提供具名
实例；`core_kernel.hpp` 和 `core_epilogue.hpp` 统一 launch 与输出映射。
12 个 WASP 编译配置包含原来未被 1178 正式用例命中的 raw record1024 配置。
正式需求历史上命中 11 个 WASP 实例和 9 个 cooperative 实例，共 20 个。
最终覆盖数量以本次正式 records 中的实际 route/legacy_instance 复核。

实例句柄是 ABI 配置编号，不是 workload case ID。运行时只使用 M/N/K/G、
stride、对齐、布局与 CU 数；phase、case ID 和目标吞吐仅用于离线报告。
预排是独立准备 kernel；未被这 1178 条选中的旧 direct kernel 为兼容保留，
不作为本轮第三套核心流水。

## 已完成检查

- 独立 cached-weight 库、benchmark bridge 编译通过；example CMake 配置通过。
- 新库 SHA256：`0052f228366c8dfb80afa9080d8227d604be6e8d404523709d3229bbe879bfb6`。
- 12 个 WASP 编译实例均无 VGPR/SGPR spill，private segment 为 0。
- packed M32/M64/M128/M256/wide N 的 VGPR/LDS 与原版一致；raw M256 的
  VGPR 从 214 降至 198/199，LDS 仍 64 KiB。这是编译资源变化，不直接等同加速。
- 旧 FP16/BF16 tag、结构体大小、cached ABI 检查通过；selector 45 项通过。
- full-range INT8 + FP64 边界 65 项 PASS：含空组、ragged、stride、尾部、
  guard、输入完全不变、预排字节、跨 stream 依赖和接口拒绝条件。
- 权重复用/刷新及异步销毁生命周期 32 项 PASS。

初次门禁在新 kernel 加载前因 HSA 初始化失败而终止；用户批准后于
2026-10-30 19:24:28 CST 重启 hymgr.service。重启后 PCI 身份
`0000:f5:00.0`（物理卡7）核实通过，以上 GPU 门禁随后通过。
原失败日志保留于 `results/refactor_v121_gates/before_hymgr_restart`。

## 全量性能回归

复测全部原始 1178 条，含 prefill 306 / decode 872，保留原 G/M/N/K。
INT8 TOPS = 2×G×M×N×K / (ms×10^9)。一轮正式测试，warmup=10、repeat=50，
三个样本取中位数；不是三轮正式测试。权重准备时间与额外存储单列。
BLAS 为此前逐 shape tuning 的冻结 Excel Max 目标，不是本轮重测 BLAS。
新结果独立输出，不能把旧 CSV、实验最快样本或新旧较优值拼入正式数据。

全输出验证采用 [-8,8] 整数输入、TF32 关闭的精确 FP32 参考、128 行分块，
检查全部输出、guard 和输入抽样不变。边界门禁另外覆盖全范围 INT8。
PASS 只描述这些验证条件，并非对任意输入穷举证明。

正式输出目录：`results/refactor_v121_all_1178`。
本轮已全部完成并核对原始 shape、完整记录和 CSV，正式结果如下。

| 范围 | shape数 | 数值PASS | R≥1 | R≥0.99 | 最小R |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill | 306 | 306 | 298 | 304 | 0.981198 |
| decode | 872 | 872 | 872 | 872 | 1.007629 |
| 总计 | 1178 | 1178 | 1170 | 1176 | 0.981198 |

无数值失败、OOM 跳过、缺失时间或吞吐。相对 v112，逐 shape speedup 中位数
为 1.000034×（+0.0034%），几何平均为 1.000323×，视为基本持平；本次整理
不宣称获得新的性能优化。未达到冻结 BLAS 目标的仍为原来 8 条 prefill：
92、96、100、704、911、920、1126、1130；差距约 0.04%–1.88%。
R=CK TOPS / BLAS target TOPS。99% 阈值附近的数量变化属于本轮实测结果，
不修改冻结目标，也不将多个版本的较好样本拼入本轮结果。

Cooperative 实际覆盖 661 条，WASP 覆盖 517 条；全部 1178 条的 route 与
legacy_instance 和 v112 一致，实际共 20 个实例。完整数据见
[current_refactor_v121_1178.csv](current_refactor_v121_1178.csv)，含 ID、phase、
M/N/K/E（G为同义列）、流水、实例、CK ms/TOPS、冻结 BLAS ms/TOPS、R、
v112 ms/TOPS、对旧版 speedup、数值状态、准备时间与权重额外存储。

单轮中 ID62（M16/N2048/K768/G16）比旧正式轮慢约 3.1%。额外的旧/新/新/旧
诊断均数值 PASS，时间中位数分别为 0.0287904 / 0.0286080 ms，未复现 3%
回退；说明该次差异未形成稳定回退证据，也不把诊断的较好数据替换正式 CSV。
诊断仅用于解释波动，原始正式值仍为新 0.0295168 / 旧 0.0286272 ms。

## 默认模板参数与二进制核验

收尾发现 M32 兼容别名的默认 A-read batch 应恢复原值 2（本轮实际实例均
显式传入 batch，因此正式运行不受该默认值影响）。修正后做默认别名语法
检查并重新构建；API 库与 benchmark bridge 均与正式测试时逐字节相同。
证据见 `refactor_v121_default_guard.json`。因此保留单轮全部正式数据，
不再重复一次全量测试。

正式运行源码 revision 是 `2a16d588b39d024e7f5646bde152fc0158bc313b`。
原运行 sources 清单保存在 `build/refactor_v121/run_sources_v121.json`，
交付 sources 清单保存在 `build/refactor_v121/sources.json`；仅上述默认参数
header 的 SHA 不同，两个 .so 的 SHA 均未改变。后来增加的构建清单工具
属于 CPU 侧复现工具，不改变 GPU 产物或正式计时数据。

构建与复测入口现可在完整 main 中独立使用：
`cached_weights/build_and_manifest.py` 负责构建并生成来源清单；随后依次
执行边界、生命周期和 `validate_all_1178.py`。实际操作命令见 example README。


## 分支与源码

- `exp/int8_nt`：保留原实验历史、实验代码及本次整理提交。
- `main`：完整 CK 源码，example 替换为两套核心流水；历史实验代码从
  main 的本 example 中移出，原许可证及源码来源保留；不导入上游 Git 历史。
- 远端工作目录和卡号保持用户指定路径与卡7，不改其它精度接口。
