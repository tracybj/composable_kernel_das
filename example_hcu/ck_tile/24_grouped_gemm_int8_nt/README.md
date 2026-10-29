# INT8 NT grouped GEMM 优化 example

本轮范围：gfx936，A/B 为有符号 INT8，累加与输出为 INT32，NT 权重 `[E,N,K]`，alpha=1、beta=0，无 scale、bias 或 zero-point。用户已于 2026-10-26 确认该口径；INT8 量化输出后续补齐。

目标为需求附录 A 的全部 1178 条 shape 正确，且每条 R=BLAS ms/CK ms >=0.5。正式一轮已完成：1178/1178条数值通过且R>=0.5，最小R=0.508321；正式报告和原始记录见results/final_01。

## 结构

- `grouped_int8_nt_direct.hpp`：小 M 的无 LDS 路径，复用 CK 的 INT8 MMAC attribute，K32 片段直接以打包向量进入寄存器。
- `grouped_int8_nt_{problem,policy,pipeline,epilogue,kernel}.hpp`：显式分层；全局内存使用 16 字节向量加载，LDS 行间 padding，下一 K tile 保持为打包寄存器；MMAC 完成后经分条 LDS 重排以 16 字节写回 INT32 输出。
- `instances.cpp`：以具名 `GroupedGemmInstanceId` 枚举映射到预编译模板实例并 launch；实例编号是公开 ABI 的内核句柄，与测试用例编号无关。
- `public_bridge.cpp`：仅负责 benchmark setup 和事件计时；热路径调用原有公开 select/query/run v1 接口。不是新的生产 API。
- `benchmark.py`：随机输入 [-8,8]，每组全输出与精确可表达整数和的 FP32 参考比较，检查输出 guard、输入抽样和 expected-instance 拒绝。
- `int8_benchmark.cpp`：既有 `19_grouped_gemm/grouped_gemm_device_args_benchmark.cpp` 的 INT8 入口；CPU 精确抽样及全量 CK/BLAS 输出/guard 对比。全量独立参考验证使用 `benchmark.py`。
- `probe.py` / `probe_bridge.cpp`：开发筛选，不作为默认 selector 验收。

公开 ABI 只增加 enum 值，原有结构体大小、字段、旧 enum 数值以及 FP16/BF16 路由保留。INT8 新值为 `DATA_TYPE_INT8_INT32_V1`，明确表示 INT32 输出，不能解释成量化 INT8 输出。

## 支持边界

INT8 首版支持 gfx936、NT、公共 N/K，256<=K<=131056 且 K%16=0，N%4=0；A/B 基址和行 stride 按 16 字节对齐，C 按 16 字节对齐且行 stride 为 4 个 INT32 的倍数。每组 M 可不同，可为空；声明的容量必须覆盖实际长度。支持正 stride，k_batch=1，无 D tensor，不支持输入输出别名。设备参数数组保持原有 GemmTransKernelArg ABI。

使用调用方 stream，run 不分配显存、不回读 GPU 长度、不做设备同步。首版 INT8 路径要求 num_cu=0；非零在 launch 前返回 unsupported，不能默默忽略。未支持的 INT8 请求返回错误，原有 FP16/BF16 回退保持不变。

## 构建

独立开发构建避免重编全部 FP16/BF16 provider：

```bash
e=/workspace/GroupGemm/composable_kernel/example_hcu/ck_tile/24_grouped_gemm_int8_nt
cmake -S "$e" -B "$e/build_cmake" \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc \
  -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936 \
  -DHIPBLASLT_INSTALL=/workspace/GroupGemm/baseline/hipblaslt-install
cmake --build "$e/build_cmake" \
  --target ck_tile_grouped_int8_nt_benchmark_api tile_example_grouped_gemm_int8_nt -j2
```

`ck_tile_hcu_grouped_gemm_int8_nt` 是仅供 INT8 开发的 ABI owner；下游应链接此 archive 或完整 `ck_tile_hcu_grouped_gemm`，不能同时链接两个 ABI owner。

完整库通过 `19_grouped_gemm/grouped_gemm_device_args_provider_sources.txt` 编译本目录的实例；根 `example_hcu/ck_tile/CMakeLists.txt` 加入本 example，gfx936 完整库启用相应公开 dispatch。配置 hipBLASLt SDK 后，既有统一 benchmark 增加 `--dtype=int8 --layout=NT` 入口。

## 测试

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7
export LD_LIBRARY_PATH=/workspace/GroupGemm/baseline/hipblaslt-install/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/workspace/GroupGemm/baseline/hipblaslt-install/lib/libhipblaslt.so.0.10
ulimit -c 0
python3 "$e/benchmark.py" --case-ids 1,6,9,35,590,598 --out "$e/results/example_pilot"
```

`--all` 才选择全部 1178 条。计时为一轮、三个 HIP event 样本取中位数，每样本预热10次、计时50次。BLAS 优先复用相同安装包、相同算法身份下已有逐 shape tuning 的有效最快算法，再同轮复测。库 SHA 或算法身份不匹配时停止，不能偷偷换目标。

`--screen` 只把新 CK 与冻结的历史 BLAS 时间比较，用于缩小优化范围，不是同轮正式验收。两种模式在 manifest 和 BLAS 状态列明确区分。

大 shape 仍使用真实 E/M/N/K 和独立专家权重，不通过共享权重、降低 group 数或分块计时替换需求。输入校验不对整个 INT8 权重一次性转换为 INT64，以免人为放大显存峰值。

结果逐例保存在 `results.csv` 和 `records.jsonl`，包含 phase、M/N/K/E、实例编号/名称、CK/BLAS ms、TOPS、R、数值与目标状态。source revision、二进制 SHA、输入口径、计时参数写入 manifest。

## 记录

设计和迭代结论见 `docs/`。`build/iteration_*` 保存开发中间二进制、编译日志、资源与 probe 日志；这些产物不提交 Git。正式交付以最终源码、公开接口回归和全量报告为准。

## 统一 C++ 入口批量复现

`cases_int8_nt_1178.csv` 与原始附录逐行对应，保留 ID/phase，包含已验证的 BLAS 算法序号。使用本次固定安装包；库SHA与算法身份见正式manifest/audit。下面的程序是原有统一benchmark源码构建的INT8入口：

```bash
python3 "$e/../19_grouped_gemm/run_grouped_gemm_device_args_cases.py" \
  --binary "$e/build_cmake/tile_example_grouped_gemm_device_args_benchmark_int8" \
  --cases "$e/cases_int8_nt_1178.csv" \
  --output-dir "$e/results/unified_replay" \
  --blas-root /workspace/GroupGemm/baseline/hipblaslt-install
```

CSV的INT8行默认使用include-args-build=0、blas-streams=1，repeat列指定每个event样本的重复次数。可先加`--dry-run`，或用`--case-ids=int8_nt_0001,int8_nt_1178`只跑部分。INT8采用TOPS列；FP16/BF16保留原TFLOPS列和原默认命令。原615行CSV的命令构造逐条对照相同，记录见docs/unified_runner_regression.json。

`cases_int8_nt_validation_1178.csv` 的time=0只用于统一入口全形状数值回归，不能作为正式性能数据。正式一轮时间和独立逐组参考验证保存在results/final_01；统一入口回归在其结束后顺序执行，避免GPU计时干扰。

统一入口的CK gfx936机器码与正式测试库逐bundle比较一致，记录见docs/unified_core_identity.json。后续增加的标准PERF/CV输出不改变被测内核。边界测试涵盖完整INT8范围；随机全形状测试和量化输出的验证口径不要混淆。

交付核对：2026-10-27（Asia/Shanghai）。统一C++入口全1178条time=0数值回归亦全部通过，0失败；该验证时间不作为正式性能。源码分流和逐实例统计见docs/PIPELINES.md，最终汇总见docs/FINAL_REPORT.md。

## Shape 驱动的选核与编号含义

生产调用链为 `M/N/K/G + dtype/layout/arch + 对齐/长度属性 → select_grouped_gemm_candidate → GroupedGemmInstanceId → 预编译模板实例 → launch`。selector 的输入结构没有 case ID 或 phase 字段。

`instances.cpp` 的 switch 负责执行已经选好的内核；例如 `Instance::int8_nt_m16_n256_aligned` 对应 16×256×64 模板。底层数值 6061 是稳定 ABI 句柄，不是需求表的第 6061 条用例。模板在编译期实例化，运行时由 shape 选择。

当前均匀组的 tile 分流主要由 M/N/K 决定。G 用于合法性检查、grid 的专家维度，以及不等长组容量摘要的 M 启发式；当前没有按 G 调整 tile 的性能规则。后续如加入总 CTA 数或小 G 分流，应以 M/N/K/G 计算特征并通过测试验证。

`benchmark.py --case-ids` 只筛选要运行的输入并关联历史 BLAS 记录；case ID 也用于报告、续跑和随机 seed。CK 只收到实际矩阵描述符。开发用 `probe.py` 可显式比较候选，不代表默认生产分流。

只在 CPU 上检查全部 shape 的选核，且核对已有正式结果中的实例：

```bash
python3 "$e/test_shape_selection.py" --results "$e/docs/selector_contract_v26.csv" \
  --output "$e/results/shape_selection_audit.json"
```

测试输入只传 M/N/K/G；还会反转输入顺序并更改测试标签后重复检查。该测试验证选核独立于 case ID，不重新测量吞吐。

## 第23轮记录

已注册3个K128/K256小M实例，按M/N/K/G与长度属性默认选核；K<1024使用16x64x128，否则G<=16用16x64x256，其余用16x128x256。旧14个实例保留，共17个INT8实例。560条小M公开入口验证见docs/iteration_23_public_audit.json；其余618条尚未完成新目标验收。早期FINAL_REPORT中的50%目标对应历史BLAS基准，本轮目标以targets_manifest.json指定Excel列为准。docs/selector_contract_v26.csv仅为1178条CPU选核契约，不能作为全量性能结果。

## 第26轮当前状态

新增18号INT8实例（ABI ID 6505）：common且无empty/ragged、M<=16、K>=4096、G<=8时用16x32x512双wave。其余小M保留第23轮规则。当前共18个INT8实例；核心dtype仍为INT8×INT8→INT32，无scale。默认入口560条验证的实际数值和达标数见docs/iteration_26_public_audit.json，其余618条新目标验收尚未完成。

## 第37轮当前状态

新增5个M32/64深K实例，当前23个INT8实例（含20个FP实例共43个注册项）。按M/K/G与common/empty/ragged属性分流，不读取caseID或phase。默认公共入口196条数值全部PASS，192/196达到固定目标；详见docs/iteration_37_public_audit.json和iteration_37_public_medium_m.csv。M<=16的分流保持第26轮规则；全1178条目标仍未完成。

## 第47轮小M公共入口验证

注册8734无LDS输出实例，uniform M<=16且K>=4096按尺寸选择，与caseID/phase无关。当前24个INT8实例，总注册44项。560/560条小M数值检查PASS，560/560达到固定Excel目标，最低R=1.007298、中位R=1.366718。28组完整INT8/FP64边界、host80+33、CPU1178维度重放PASS。详见docs/iteration_47_public_audit.json及CSV。本轮不代表全1178达标，大M及中等M剩余差距继续优化。

## 第73轮公共入口集成

新增三个独立WASP实例（19001/19002/19003），统一非空大M按尺寸选择，宽stride等请求保留kernel内通用fallback。Host38+80、CPU1178重放/改标签/倒序、34项公共全范围边界检查PASS。新库全部1178条默认性能测试尚待完成；旧版及oracle数据不作为本版验收结果。详见docs/iteration_73_public_integration.md。

## 第80轮公共 record-stride 策略

在现有 WASP problem/pipeline 中增加可选 RecordCap0/1024，新增稳定实例19004/19005，非幂次 K 按 M/K 维度分流。原有数学、per-group stride、native guard 和 packed fallback 保留。Host45+80、CPU1178重放/改标签/倒序、44项公共全范围边界检查PASS；本版1178条正式性能待测，v73正式结果保留。详见docs/iteration_80_public_record_policy.md。
