# INT8 NT 带scale：Max ALL目标迭代记录

## 固定目标与验证口径

- 来源：用户提供的`BlasLt_ggemm_tuning_baseline.xlsx`第三页`int8_NT`，P3:P1193，共1191条。文件SHA256：`adbb420c4e4d9a3a7a30520c8af55369b94ae79f54ba3b8c0410c1fe2a6478ae`。
- P列实际标题`Max ALL Tops`，每行公式为`MAX(K行,O行)`；K为INT32最优，O为带scale最优。已逐行检查缓存值与公式输入一致。只提取所需shape/目标，原工作簿不修改、不提交。
- 原1191条ID/M/N/K/G逐条匹配。后13条phase按本次工作簿更新为9条decode、4条prefill；总计881条decode、310条prefill。旧版归档中的supplemental标签不改写。
- 实现仍是INT8×INT8、INT32累加、group标量激活scale、per-channel权重scale，在kernel内反量化写BF16或FP16。原无scale INT32接口及其它16位输入接口保留。
- P列作为固定的工程吞吐目标。它来自不同BLAS模式的最大值，不代表原生BLAS与本实现的输出/scale语义完全相同。
- `R = CK_TOPS / P目标`；严格达标为数值校验PASS且`R>=1`。`R>=0.99`仅作波动观察，不能替代严格达标。TOPS按`2GMNK/(ms×10^9)`计算。
- 权重准备只做一次并复用，时间/额外存储单列，GEMM事件计时不含准备。每轮保留原始样本、库SHA、编译选项、完整输出验证和失败原因。正确性不放宽。
- Runtime分流仅由M/N/K/G、dtype、stride/对齐及设备资源决定，不读取case ID、Excel或目标表。

目标提取见[1191条目标](targets_max_all_1191.csv)、[来源及基线统计](max_all_target_manifest.json)。

## v123：本轮起点

代码`e565e9dd`，结果归档`ac59e6b`。已在卡7完成两种输出各1191条完整校验。

| 输出 | 严格达标 | R>=0.99 | R>=0.5 | 未严格达标 | 最小R |
|---|---:|---:|---:|---:|---:|
| BF16 | 1093/1191 | 1103/1191 | 1189/1191 | 98 | 0.4780 |
| FP16 | 1108/1191 | 1128/1191 | 1191/1191 | 83 | 0.5913 |

BF16剩余98条按route计：route12为62条、route0为13条、route5为9条、route7为8条、route3为3条、route4为2条、route1为1条。最大差距是N256/K512/G64下M256、2048、4096三条，之后的未达标项大多在目标的91%以上。完整差距见[max_all_v123_baseline.csv](max_all_v123_baseline.csv)。

## v124：现有实例受控筛选（进行中）

### 假设和方法

先检查实例选择是否已限制性能，而不是直接重写第三条主流水。

1. N256/K512低算术强度族目前在M256/2048/4096使用逻辑转置M256×N256 WASP，LDS64KiB、4个producer wave和8个consumer wave。候选为现有packed M128×N256/M128×N128及较小Cooperative tile；评估更低LDS、更少producer、连续16位写回与并行任务数量的组合收益。
2. 大M窄差距族比较现有packed几何/方向，先测代表shape和相邻尺寸；若现有实例均无收益，再设计有数据流依据的新实例或主循环。
3. 小M剩余项独立分析，不能用大M的选择规则覆盖；同时保留已达标样本作回归对照。

新增隔离构建开关`CK_INT8_CACHED_TUNING`，默认OFF。仅在ON的实验库中，通过`CK_INT8_TUNE_ROUTE`与`CK_INT8_TUNE_LEGACY`选择预编译实例，且在launch前检查uniform/native、完整tile、stride和packed-weight约束。环境参数只用于实验，不进入默认部署选择。

构建示例：

```sh
python3 cached_weights/build_and_manifest.py --build build/max_all_v124_tune --tuning --jobs 2
```

### 本轮结果

尚未完成，待编译、数值和性能筛选后补充。负结果同样保留，未验证候选不改默认分流。

首轮筛选在ID1188进入测试数据准备时异常退出。首个故障符号为Torch的`index_elementwise_kernel`，而非GEMM。原因是新测试脚本用FP32 `linspace`产生大tensor采样索引：`64×2048×512-1=67108863`在FP32中舍入为`67108864`，成为越界位置。已将采样改为INT64整数等距计算，并在取样前检查边界；失败原始目录`results/max_all_v124_low_n`保留，使用新目录`results/max_all_v124_low_n_r2`复测。没有据此将GEMM候选记为失败，也没有重启共享服务。

退出前154项已完成候选完整输出校验无错误。M256/N256/K512/G64的暂选M64×N256 packed WASP（route8）BF16约118.8 TOPS、FP16约120.9 TOPS，均高于84.96目标；仍需独立正式复测后才能写入默认几何分流。

整数采样修复后的r2筛选已完成15个shape、382项有效候选输出检查，完整校验错误0。初筛为3次预热、10次重复、2个样本，尚非最终验收计时。

| M/N/K/G | 暂选实例 | BF16 TOPS / 目标 | FP16 TOPS / 目标 | BF16相对默认加速 |
|---|---|---:|---:|---:|
| 256/256/512/64 | packed M64×N256，route8 | 117.45 / 84.96 | 119.60 / 84.96 | 2.34× |
| 2048/256/512/64 | packed M128×N256，route6 | 194.12 / 139.33 | 191.06 / 139.33 | 2.80× |
| 4096/256/512/64 | packed M128×N256，route6 | 204.97 / 146.97 | 203.86 / 146.97 | 2.92× |

同族M8192/16384的默认route6仍是最快或处于波动内；新增邻点M512/1024暂选route8。下一步独立使用10次预热、50次重复、3样本中位数，覆盖G4/16/32/64/128及相邻M，检验基于CTA数量而非case ID的分流边界。之后再比较大M剩余代表形状。

筛选明细见[max_all_v124_low_n_screen.csv](max_all_v124_low_n_screen.csv)。实验库SHA256为`988ed463158d24d1b8c0a3bf005cf035d40f85d4563d633a67bb98cab7a89766`，默认生产选择尚未改动。

## 分支与提交

本地与服务器基于`ac59e6b`创建`exp/int8_nt`，main保留v123。每轮用明确说明改动的commit提交方法、结果与必要实现。GitHub连接通过当前系统代理恢复；已正常推送main的v123归档和实验分支起点`2364ef5`，保留远端原实验历史，没有强推。

## 后续数据路径对照（用户补充，尚未实施）

在现有Cooperative/WASP两套主流水内，比较A/B都经LDS、仅A经LDS、仅B经LDS、A/B都直入寄存器的实例。先覆盖小M、短K、低N和对应已达标回归样本。

当前Cooperative的LDS负责物理连续K方向加载后的MMAC寄存器分发，并提供跨wave复用。wave内shuffle不能直接取其它wave的寄存器；直读方案需明确重复GMEM请求、向量宽度、缓存流量、VGPR和有效驻留的代价。不能以LDS占用减少直接判断成功。

WASP原方向已有A走LDS、B走寄存器的实现，逻辑转置方向可交换原操作数角色；这些是可借鉴的已有数据路径，不把它们未经测量推广为所有Cooperative几何的最优解。保留数值/scale/输出规则，先比较完整kernel时间，再用ISA/资源和受控消融解释差异。
