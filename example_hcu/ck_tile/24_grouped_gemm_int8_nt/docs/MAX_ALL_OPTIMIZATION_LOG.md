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

### v126结果与边界

API库SHA256为`057cff2a38793f61c621c0bb38e402403a716536d670c0ca002a8a935aa9de44`，代码7f387d7。486项full-range INT8/FP64参考完整输出门禁全部PASS，无prelaunch拒绝；BF16/FP16同轮正式对照19个shape、874项完整输出检查均PASS，10次预热、50次重复、3样本中位数。

连续16-byte+shuffle能够改善部分同几何候选，但未超过已测shape的全局默认最优。例M32/N2048/K512/G64，在BM16/BN128/BK256下：双LDS为23.444 TOPS、B按8B直取22.825、A LDS+B16B/shuffle为34.733（同tile提升48.1%）；现有另一BM16/BN64/BK128双LDS实例仍为41.684 TOPS。不能以跨tile比较否定单边LDS，也不能把48%收益报成当前默认性能提升。

ISA已确认8-byte候选使用global_load_dwordx2，shuffle候选使用global_load_dwordx4。BM16/BN64/BK128静态函数中，仅A LDS+shuffle相比双LDS，ds_write_b128为2比10条，但新增32条ds_bpermute；双寄存器+shuffle仍有输出重排的LDS和barrier。静态指令数量包含prologue/steady/epilogue，不能当成动态执行次数。该组元数据见[max_all_v126_isa_summary.json](max_all_v126_isa_summary.json)，性能明细见[max_all_v126_shuffle_formal.csv](max_all_v126_shuffle_formal.csv)。

## v127：延后shuffle，恢复下一块加载的计算重叠

源码复查发现v126的shuffle放在Load(next)内部，随后的sched_barrier把当前MMAC隔开；shuffle可能迫使下一块VMEM提前等待完成，削弱原Cooperative的load-next/compute-current重叠。不能只将结果归因于shuffle指令数量。

新增9022/9023/9024，与9013/9016/9019保持同tile/同搬运分布，仅把下一块16B寄存器重排延后到当前块MMAC之后；prologue先重排当前块。9031/9032分别检验BN128/BK128和BN64/BK64的A-LDS/B-shuffle几何。9034/9035/9036为新Problem特化下A/B双LDS对照，检查通用模板本身是否引入额外差异。所有候选仍属Cooperative家族、实验构建；先做full-range/尾部/scale门禁，再同轮计时。未取得结果，不改生产选择。

### v127结果

代码7084458，隔离库`build/max_all_v127_deferred`的API SHA256为`039e98a2fc2f337de02bea38a2a16580e4543b665598cbcbaca931409018ac91`。432项full-range INT8/FP64参考完整输出门禁全部PASS，0拒绝；19个shape、608项同轮正式记录完整输出PASS，0错误。计时为10次预热、50次重复、3样本中位数，无其它卡7实验并行。

| BF16测试shape M/N/K/G | 相同几何双LDS | 即时shuffle | 延后shuffle | 现有默认最优 |
|---|---:|---:|---:|---:|
| 32/2048/512/64；BM16/BN128/BK256 | 23.432 | 34.662 | 36.061 | 41.885 |
| 1/4096/7168/8；BM16/BN64/BK256 | 2.137 | 1.593 | 1.787 | 2.319 |
| 1/6144/7168/12；BM16/BN64/BK256 | 2.290 | 1.312 | 1.795 | 2.424 |

以上单位均TOPS，全部取自同轮记录。第一行延后shuffle相对同几何双LDS提升53.9%，相对即时shuffle提升4.0%；第三行延后shuffle相对即时版提升36.8%，但仍落后原双LDS。说明重排时机是重要因素，且去掉LDS并非通用收益。

新增BN128/BK128及BN64/BK64也未超过当前已测样本的全局最优。9034/9035/9036只是原双LDS的模板对照，部分以约1%波动成为计时最优，不视为新数据路径的收益。BF16/FP16下这三组模板对照的完整kernel机器码与对应原实例逐字节相同（6/6），排除了新Problem封装本身引入代码差异的混杂因素。

实际ISA中，即时版在当前块MMAC前出现下一块VMEM等待及shuffle；延后版的steady-state MMAC可位于该等待/重排之前。BM16/BN64/BK128静态VMEM/shuffle/MMAC条数不变，VGPR为69对即时70，LDS仍4096B；不能把计时变化仅归因于静态指令数。控制码等价及ISA摘要见[max_all_v127_control_equivalence.json](max_all_v127_control_equivalence.json)、[max_all_v127_isa_summary.json](max_all_v127_isa_summary.json)，完整记录见[max_all_v127_deferred_formal.csv](max_all_v127_deferred_formal.csv)。分支内保留所有候选，默认生产选择不变。

### 操作数路径的当前判断

- 双LDS：连续向量搬运后共享分发，适合跨wave复用；代价是LDS读写和barrier。
- 单边LDS：有明确的同几何正收益，应作为Cooperative的搬运策略参数继续研究。优先保留跨wave复用多的一侧，另侧在本wave内完成加载/重排；不能只根据原始M小或大硬切。
- 双寄存器：去掉操作数LDS和相关CTA barrier，仍有输出LDS；本轮8B/16B实现均未胜过现有最优，且存在跨wave重复加载、shuffle和VGPR代价，暂不推广。
- shuffle只能在wave内交换，不能替代跨wave共享。WASP原方向已是A-LDS/B-register，逻辑转置可交换原始操作数角色；不需要新增第三套核心流水。
- 后续继续以M/N/K/G、tile/wave布局、stride和准备好的权重布局选择实例；不按case ID或目标表决定运行分流。本轮仅代表shape路径对照，不能报告为1191条全量改进，更未宣称达到硬件上限。

### 补归档v124已完成的正式对照

低N边界正式对照27个shape、270条记录全部完整输出PASS；N2048/K768族16个shape、128条记录全部完整输出PASS。均为10次预热、50次重复、3样本中位数。此前仅r2初筛表已提交，本次补存正式CSV，避免与实验结论丢失。

N256/K512/G64，M256选route8为119.336 TOPS（目标84.961）；M2048选route6为194.079（目标139.329）；M4096选route6为205.879（目标146.975）。N2048/K768、M>=2048的16条选route6，BF16/FP16均超过固定目标，最小R约1.0329。相邻G测试及负结果保留在[max_all_v124_low_n_formal.csv](max_all_v124_low_n_formal.csv)、[max_all_v124_k768_formal.csv](max_all_v124_k768_formal.csv)。这两类已有候选可供后续独立的scaled selector改动及全量回归使用，本轮仍未修改默认selector。

## v128：接入已验证的scaled几何选择，保留INT32策略

v124正式对照已证实两类实例收益，本轮将其接入生产构建，而非停留在环境变量实验：N256/K512、M256–1024选择packed M64×N256（route8），M>=2048选择packed M128×N256（route6）；N2048/K768、M>=2048选择route6。选择只依赖尺寸及packed可用性，仍由原native/uniform/stride域保护；其它形状沿用原策略。

为在生成kernel参数前决定raw/packed指针，把原plan构造提取为内部make_plan，v1传INT32、v2传实际输出类型；不先创建旧plan再重分配参数。9个公共ABI函数及结构不变，无scale INT32继续使用原selector，其它BF16/FP16输入算子不修改。

新增host policy断言及三组production API数值用例（NULL scale、scalar/per-channel/per-token、BF16/FP16及INT32旧route），同时复跑边界/生命周期/epilogue门禁。通过后在卡7对1191条按Max ALL固定目标跑一轮正式计时、两种输出完整校验；最终R>=1才计达标。编译和全量结果待补。

### v128门禁及全量启动

实现提交32027d2；生产构建`build/max_all_v128_scaled_policy`，实验开关OFF，API SHA256为`270194f6808e60f58b4cc6e78bc566b648ee8c7a18685753d309d697348dc92c`，公共导出仍9项。host分流24项断言PASS；39配置355项完整输出scale门禁、7项无效参数拒绝、6项原生BLAS标量scale逐位对照PASS；native epilogue18项、原INT32边界65项、生命周期32项PASS。

两个设备代码对象与v123均逐字节相同，SHA为`687e11d7...`和`a2255eb4...`，完整值见[max_all_v128_validation_summary.json](max_all_v128_validation_summary.json)。本轮收益来自按输出类型选择已有实例；未改计算设备码，也未让新scale选择覆盖原INT32策略。

随后启动`results/max_all_v128_full_1191`的一轮全量BF16/FP16回归，代码和目标快照冻结。使用Max ALL P列1191条原始M/N/K/G及881 decode/310 prefill；每条10次预热、50次重复、3样本中位数。每组完整输出参考按128行分块检查，另查guard及输入/scale抽样不变，失败保留现场。新增正式CSV直接包含BLAS目标时间/TOPS及双输出R，不用后续最快样本拼接正式数据。正在运行，尚不更新全量达标数。

### 对v125大M驻留假设的核实

只在卡7、无其它本任务GEMM计时运行时，加载已测代码对象并调用HIP module occupancy计算API。实际设备报告80个CU、wave64、每block最多1024线程、每CU最多2560线程、LDS每CU64KiB。regsPerBlock为196608，不能把这个字段直接当成per-CU寄存器总量。

| packed WASP BM/BN | 线程 | VGPR | LDS bytes | API每CU最大CTA |
|---|---:|---:|---:|---:|
| 256/256 | 768 | 199 | 65536 | 1 |
| 256/384 | 1024 | 183 | 65536 | 1 |
| 128/256 | 640 | 119 | 32768 | 2 |
| 128/128 | 384 | 135 | 32768 | 2 |
| 64/256 | 576 | 87 | 16384 | 3 |
| 128/384 | 896 | 119 | 32768 | 1 |

v125 BM128/BN384虽把LDS减半，资源组合下API最大驻留仍只有1 CTA，未满足原假设的2 CTA前提；这会改变后续设计，不能继续仅按LDS估计驻留。该查询是静态可行上限，不是动态achieved occupancy，也不能单独证明性能瓶颈。数据及精确Problem参数见[max_all_wasp_occupancy.json](max_all_wasp_occupancy.json)。

## v129草案：先反量化打包，再交换16-bit输出

当前native WASP先对四个INT32累加值做4次ds_bpermute，再逐个反量化写16-bit。本候选不改主循环：使用原MMAC坐标对应的8个通道系数，先调用同一个Int8NtDequantConvert（相同FP32乘法和RNE/NaN规则），将4个16-bit结果打包为两个32-bit word，再用两次相邻quad交换加一次跨pair交换形成连续4列写回。每四输出的wave交换由4条变为3条，但新增半字打包/选择，不能按指令减少比例预报kernel加速。

草案位于experiments/packed_output_v129/epilogue.hpp，仅原方向native/scalar-A路径候选；逻辑转置和per-token沿用Parent，INT32无此epilogue。test_layout.py在CPU完成4105组quad场景的位级映射证明，包括随机16-bit及NaN/Inf/符号位模式；这不是GPU数值或性能证明。

目前未接入kernel模板/dispatch、未编译、未运行GPU，不改变v128全量使用的源码及二进制。待本轮全量完成后，使用相同几何/主流水成对验证，再决定保留或放弃；不以缩小数值契约换取收益。
