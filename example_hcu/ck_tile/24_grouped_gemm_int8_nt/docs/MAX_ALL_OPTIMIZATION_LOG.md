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

## v125：Cooperative操作数路径与WASP驻留候选（准备验证）

v124代表形状筛选显示：K768/N2048族可由现有packed M128×N256改善约13–14%，但大部分其余prefill最优仍是现有route12，距目标约2–7%；小M代表族现有实例的最佳收益也不足以补齐目标。该负结果仅针对已测几何，不据此认定硬件上限。

### Cooperative数据路径

实验类型继承现有Problem，特化Policy和Cooperative的load-next/compute-current阶段，复用原Kernel与反量化epilogue；不是新增第三种producer/consumer调度。

- 对照为原A/B均走LDS的实例；私有9001–9003为BM16/BN64/BK128下仅A LDS、仅B LDS、A/B操作数都直入寄存器；9004–9006对应BK256；9007–9009对应BM16/BN128/BK256；9010–9012对应BM128/BN128/BK64。
- 首个直读候选按MMAC要求直接读取每lane的8-byte片段，不先强制加入shuffle。代价是较窄VMEM请求、A按WN和B按WM的跨wave重复加载；后续若请求开销占主导，再比较连续16-byte读取后wave内shuffle的配对方案。
- 混合路径仍保留LDS读完后可覆盖、下一stage可读的两次跨wave同步。两操作数都私有时去掉主循环的操作数barrier；输出epilogue仍有LDS重排，不能把它描述为整个kernel完全不使用LDS。
- 默认Problem及生产构建不变，这些私有实例只在`CK_INT8_CACHED_TUNING=ON`的隔离库编译、选择。

### WASP几何与驻留

增加BM128/BN384 packed候选，复用同一WASP流程。相比BM256/BN384，A双缓冲LDS从64KiB降到32KiB，producer从4个wave降到2个，consumer仍12个；假设两个CTA的实际驻留能改善MMAC/搬运延迟隐藏。代价是B请求与CTA数量增加，必须用实测资源与时间判断。

私有route13/14为A读取batch2/4、M tile cluster4；route16/17把cluster改为8作配对对照。route15为BM128/BN256、batch4，用于区别宽N与batch变化。尚未取得编译、资源和数值/性能结果，不进入默认选择。

### 编译、资源和数值门禁

隔离库`build/max_all_v125_paths`编译通过，API SHA256为`b73e8fe8e636ad1ce3b64fc153dc059346d21d9310842fb5581c54bad68c1920`。全范围INT8/FP64参考的783项完整输出检查通过；另外135项不满足WASP native域的选择在launch前被拒绝。覆盖INT32/BF16/FP16、NULL/per-channel/per-token scale、NaN/Inf/零、M/N/K尾部、padding和输入/输出guard。

实际资源元数据：BM16/BN64/BK128仅A LDS和双寄存器均占4096B LDS、约66 VGPR；仅B LDS占9216B、约78 VGPR。BK256双寄存器约114 VGPR，说明减少LDS可能增加寄存器压力；C重排的LDS仍保留。M128/BN384 WASP batch2为119 VGPR、batch4为135 VGPR，LDS均32768B。所查新候选均无spill/private segment。没有把这些静态资源直接当作性能结论，下面继续实测。

### v125完整kernel筛选结果

操作数路径筛选完成19个shape、646项有效输出检查，完整输出校验0错误；宽N WASP筛选完成16个shape、256项检查，8行/group采样校验0错误。初筛均为3次预热、10次重复、2样本，不作为全量最终验收。

在本轮19个小M/回归样本中，9001–9012没有一项成为最快候选。例：M1/N4096/K7168/G8，原最优2.322 TOPS，BM16/BN64/BK256仅B LDS为2.053、仅A LDS为1.446、双寄存器为1.311。M32/N2048/K512/G64，原6401为41.67，仅B LDS为31.45、仅A LDS为22.50、双寄存器为16.37。该结果否定的是当前8-byte直读组合的收益，不能推导寄存器直取整体无效；窄VMEM和跨wave重复请求的相对贡献尚未独立测量。

新BM128/BN384 WASP route13–17也未在16个代表prefill中获胜，减少LDS/producer不足以抵消额外CTA/权重加载等代价；动态驻留和请求代价尚未采counter。原route12仍主导大M。所有新路径仍只在实验构建启用，生产选择不变。

明细见[max_all_v125_operand_screen.csv](max_all_v125_operand_screen.csv)、[max_all_v125_wide_screen.csv](max_all_v125_wide_screen.csv)。这两项均使用API库b73e8fe8，完整SHA见上述编译门禁，原始samples/manifest分别保存在同名results目录。

## v126：连续16-byte直读与wave shuffle配对实验

保持Cooperative调度和同一BM/BN/BK，增加9013–9021，与9001–9009逐项配对。每64B K stripe，四组lane分别连续加载16B，再用两级butterfly、共4条ds_bpermute交换4个word，产生两次MMAC所需的8B/lane fragment。CPU映射证明覆盖四个lane quad；GPU全范围数值门禁和独立计时尚待执行。shuffle使用wave交换指令，不分配操作数LDS，但仍消耗交换指令及寄存器，跨wave重复加载仍在，输出LDS也保留。

反证条件：若新向量化shuffle仍落后原双LDS，则本轮不推广此路径，保留负结果，再从任务划分、加载复用或预排布局寻找证据。没有以“省LDS”为成功标准。
