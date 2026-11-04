# INT8 NT scale epilogue v123

2026-11-04，ck_yyc / 物理卡7 / gfx936。构建目录`build/scale_v123`；旧库保留在`build/scale_v122`。

本轮仅优化INT8到BF16/FP16的输出尾部。Cooperative/WASP主流水、实例选择、原有无scale INT32入口、其它16位输入的接口均保持原有实现。

## 实现

- native WASP整块路径对标量激活scale单独处理。通道scale与激活scale的FP32乘积在epilogue中预载并重复使用；支持标量/空权重scale。系数每次kernel调用都重新读取，不跨调用缓存scale值。
- 原向输出方向每个lane的8个通道系数跨M行复用；逻辑转置方向每个lane按输出N坐标预载系数，供对应的多个输出M位置复用。scale坐标始终对应原始输出矩阵。
- native路径依靠已有整块/32列consumer条带约束，去掉逐输出边界判断；BN384最后一个tile仍检查consumer条带边界。per-token激活scale及非native域保留通用路径。
- BF16使用无分支RNE/NaN选择；FP16继续保留FP32结果的舍入边界。通用Cooperative尾部也复用每4个输出共同的激活scale。
- 数值顺序仍是`FP32(acc) * FP32(scaleA * scaleB)`再按输出精度RNE，不放宽精度或特殊值规则。

## 数值门禁

新库API SHA256：`f2f4b18d6e8022acaefab0c8c7296164f1c625ae47fbab61b335931bbebe185b`。

- 36个配置、328项完整输出scale门禁通过，包括全范围INT8、FP64整数参考、标量/per-channel/per-token、NULL、舍入、特殊值和补充13条shape。
- 与安装的BLASLt原生标量scale结果进行的6项逐位对照通过。
- 新增`test_scaled_epilogue.py`的18项native完整输出检查通过：packed、逻辑转置、BN384非整tile consumer尾部，NaN/Inf/正负零，以及同一plan反复更新scale。
- 原有65项INT32边界/接口/stream/权重生命周期检查通过。
- 原始日志在`results/scaled_v123_gates`，验证摘要见`scale_v123_validation.json`和`scale_v123_epilogue_validation.json`。

## 关键用例受控复测

同一库中分别走无scale INT32与带scale16位输出；预热10次，重复50次，3样本中位数，准备时间不计入。所有测试均在卡7。v122对照来自同日此前的受控诊断，完整原始样本另存。

| ID | M/N/K/G | v122 BF16+scale ms | v123 BF16+scale ms | v123 FP16+scale ms | 同轮INT32 ms |
|---|---|---:|---:|---:|---:|
| 25 | 128/2048/512/16 | 0.085946 | 0.041133 | 0.040605 | 0.043536 |
| 36 | 4096/2048/512/64 | 9.258763 | 2.301331 | 2.177587 | 2.868959 |

上表是定位阶段的小样本：ID25做完整输出检查，ID36检查每组8个行位置。下节给出本轮1191条正式性能和完整输出回归，包含这两条。

## 1191条正式回归

实现提交`e565e9ddb7cad27536f0fa2f33f74a7ef2f332ee`。同日在卡7完成1191条原始shape各BF16/FP16两种输出的单轮回归：**各1191条PASS，失败0、跳过0**。完整参考输出使用[-8,8]输入、关闭TF32的精确整数FP32点积，再按既定scale乘法顺序和输出舍入规则比较；guard及输入/scale抽样不变检查通过。全范围INT8参考另见前述数值门禁。

每条M/N/K/G及phase与原清单逐项一致，包含306条prefill、872条decode、13条supplemental。逐条复算3个计时样本的中位数及`2GMNK/(ms×10^9)` TOPS，并核对双输出合并表和两份单输出表一致。原始记录、运行环境、源码快照和日志位于`results/scaled_1191_v123_20261104`。

| 输出 | 全量中位加速 | 全量几何平均加速 | prefill中位加速 | decode中位加速 | 比旧版慢超过5%的条数 |
|---|---:|---:|---:|---:|---:|
| BF16 | 1.046× | 1.245× | 1.712× | 1.027× | 0 |
| FP16 | 1.036× | 1.226× | 1.650× | 1.021× | 0 |

加速比均为旧scale版v122时间 / v123时间。两版均在同日卡7按相同输入seed、scale模式、权重复用、预热/重复/样本参数测试，但不是同进程交错测量；几个百分点的差异不作强归因。权重准备时间和额外存储仍单列，不计入GEMM；没有新增全局INT32中间结果或独立反量化kernel。该比较不含BLASLt带per-channel scale性能。

正式全量记录中的两条重点用例：

| ID | 旧BF16 ms | 新BF16 ms / TOPS | 旧FP16 ms | 新FP16 ms / TOPS |
|---|---:|---:|---:|---:|
| 25 | 0.085114 | 0.040989 / 104.78 | 0.080950 | 0.039626 / 108.39 |
| 36 | 9.323416 | 2.353065 / 233.63 | 8.020005 | 2.198989 / 250.00 |

无scale INT32设备代码对象与v122逐字节一致，SHA256为`687e11d707a065fe3b5b17f884dd55e3c1c9776913966bc9807215859ef04fa9`。两个重点实例VGPR/LDS不变且无spill，整库scaled kernel元数据均无spill/private segment，见[设备代码摘要](scale_v123_codegen_summary.json)。

## 与INT32的剩余差距

以已有v121无scale INT32全量结果作历史参照，新BF16/FP16中位耗时比约0.999/0.996，整体已接近。该历史参照不是本轮全量交错复测，不能据此对每条精确归因。

仍有11条BF16、7条FP16比历史INT32慢超过20%，最大约31%，主要是较小M的Cooperative路径或非native域；这轮重点的WASP scale复用不覆盖所有通用尾部成本。因此不声称1191条全部与INT32等速。对照CSV单列`INT32_v121_reference_ms`及相应时间比，后续可对这些用例做同轮INT32确认，再优化Cooperative的scale复用和动态判断。

- [1191条新版双输出CSV](scale_v123_1191.csv)
- [1191条旧新版及历史INT32参照CSV](scale_v123_vs_v122_1191.csv)
- [性能统计及剩余用例ID](scale_v123_performance_summary.json)
