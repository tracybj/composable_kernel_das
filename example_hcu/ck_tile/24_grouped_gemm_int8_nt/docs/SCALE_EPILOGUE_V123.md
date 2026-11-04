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

该小样本中ID25做完整输出检查，ID36检查每组8个行位置；两者的带scale输出已包含在上一轮1191条完整输出校验中。不能据这两条宣称所有shape与INT32等速；完整1191条性能及完整输出回归另行归档。
