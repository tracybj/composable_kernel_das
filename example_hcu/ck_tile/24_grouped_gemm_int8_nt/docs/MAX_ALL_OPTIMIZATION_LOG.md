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

### v128全量结束及审计

同一生产候选库270194f6...于卡7完成一轮1191条正式回归，BF16/FP16各1191条完整输出PASS，失败0、跳过0。已逐条核对Excel冻结目标的M/N/K/G和phase、单轮记录、3样本中位数、2GMNK TOPS、R、guard、输入/scale抽样不变、合并表及两份单输出表；运行结束的源码快照与开始一致。不是将初筛最快样本拼接成全量。

| 输出 | 严格R>=1 | R>=0.99（仅观察） | 未严格达标 | 最小R | prefill严格达标 | decode严格达标 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1109/1191 | 1119/1191 | 82 | 0.92335 | 241/310 | 868/881 |
| FP16 | 1123/1191 | 1142/1191 | 68 | 0.94061 | 255/310 | 868/881 |

对比v123的1093/1108条，当前全量分别增加16/15条严格达标，没有旧已达标项掉到1以下。19条的route实际改变；其中13条BF16、12条FP16从未达标变为达标。其余跨线发生在route未变的样本，不能将这几条的测量波动归因于selector改动；两轮全量不是交错计时。完整route变化和跨线ID见[max_all_v128_policy_changes.json](max_all_v128_policy_changes.json)。

| shape M/N/K/G | BF16 TOPS | BF16 R | FP16 TOPS | FP16 R |
|---|---:|---:|---:|---:|
| 256/256/512/64 | 119.220 | 1.4032 | 121.048 | 1.4247 |
| 2048/256/512/64 | 195.258 | 1.4014 | 191.493 | 1.3744 |
| 4096/256/512/64 | 205.962 | 1.4013 | 204.392 | 1.3907 |
| 2048/2048/768/4 | 277.415 | 1.0449 | 276.576 | 1.0417 |
| 16384/2048/768/4 | 315.232 | 1.0711 | 311.868 | 1.0596 |

BF16剩余69条prefill、13条decode；FP16剩余55条prefill、13条decode。当前最差BF16为M4096/N5120/K2304/G48，497.254 TOPS、R0.92335；N3072/K4096等大M族仍需约数个百分点改进。不能把全量数值PASS称为全部性能达标，更不能将R>=0.99计入严格完成。

归档：[1191条完整双输出CSV](max_all_v128_1191.csv)、[任一输出未达标清单](max_all_v128_remaining.csv)、[审计和统计](max_all_v128_summary.json)。原始3样本、manifest、单输出CSV及完整日志位于results/max_all_v128_full_1191；源码实现32027d2，后续提交仅增加文档和未启用实验草案。原无scale INT32策略及两个设备代码对象均保持，未改其它16位输入接口。

### v129草案补充检查及下一步

已对原方向四种几何、BF16/FP16及逻辑转置fallback进行gfx936编译器语法检查，返回0；仍未构建完整GEMM候选库、未执行GPU数值/计时，因此不报收益。先按相同主流水/相同tile对照该输出交换方案。

小M的13条未达标集中于M1/2、长K。此前单边LDS没有跨过当前最优，但那组B的padding仍占用额外共享内存。下一可验证组合是A寄存器+B紧凑XOR LDS：BM16/BN32/BK512下仅B需16KiB（原双LDS约25KiB），假设驻留上限可从2变4；这是尚未实现的资源/访存假设，需要先做逐byte映射、实际occupancy及全范围数值门禁，再测VMEM请求/时间。不能因单边LDS旧候选失败而排除这个组合，也不能以省LDS预报性能。

### v129接入隔离构建

为ScaledWaspKernel增加默认值为原epilogue的可选类型参数，原有调用仍使用原实现；实验route18/19/20/21分别对照已有12/5/6/7的相同Problem/tile，只替换输出重排。INT32调度仍调用同一个原kernel。新epilogue头文件加入tuning构建的source hash集合，避免验证遗漏实际依赖。未改默认selector，等待编译、资源和GPU数值/计时。

## v130：A寄存器+B紧凑XOR LDS，检验小M实际驻留

在既有Cooperative实验Problem/Policy上增加Compact策略；K片段映射为k XOR ((local_row&7)*16)，去掉每行16B padding，保持连续16B GMEM搬运及完整的8B MMAC fragment内容。静态限制BK>=128，避免XOR越出短K行。CPU逐byte写入/读取证明通过184320个byte检查、8个完整tile，无覆盖或遗漏；这不是bank-counter或GPU性能证明。

9041/9042为BM16/BN32/BK512仅B走紧凑LDS（A8B直取/16B延后shuffle）；9043/9044对应BN64/BK256，9045/9046对应BN128/BK256；9047/9048/9049保留两操作数紧凑LDS，作为去padding与A旁路的消融对照。9项都是实例ID，不是case ID，仍仅在实验开关ON编译/选择。

预测：BN32/BK512仅B占16KiB，BN128/BK256仅B占32KiB；需用实际寄存器与HIP occupancy计算验证最大CTA数是否由2→4、1→2，再看完整kernel。混合路径保留读完成/发布的同步，输出C重排的LDS仍存在。候选有可能因A重复加载、地址算术或bank访问变慢；反证成立即不推广。全范围INT8/FP64、guard、tail/padding、scale与两种16bit输出门禁通过后，做同几何对照，再决定下一步，不启动并行卡7任务。

### v129/v130结果

源码15663b9，隔离库build/max_all_v130_output_compact，API SHA256为924dc278f39b3f9509d228a56218e0f4bce7864db7919520d2e649f40f60c6ab。576项full-range INT8/FP64完整输出门禁PASS，126项不满足WASP/pack域的配置在launch前拒绝。整库261个kernel的资源元数据无private/spill。四种packed-output几何的VGPR/LDS峰值与各自对照相同。

v130紧凑LDS正式对照19个shape、608项完整输出检查PASS，10 warm/50 repeat/3样本中位数。HIP API确认：BM16/BN32/BK512仅B紧凑LDS16KiB，最大4 CTA（旧25KiB、2 CTA）；BN128/BK256仅B32KiB，最大2 CTA（旧约39KiB、1 CTA）。但没有填平13条小M目标：例如M1/N4096/K8192/G64，原默认2.3424 TOPS，原同tile6404为2.3641，紧凑9045为2.3803（R0.9680）；M1/N6144/K7168/G12默认2.4240，紧凑B512为2.3633，仍回退。只能说部分shape约1–2%改善，不能把静态最大驻留翻倍解释为已达到动态驻留或性能翻倍。没有推广。

v129输出重排初筛22个shape、396项输出检查PASS；参考每组8行，3 warm/10 repeat/2样本，非全量验收。对照为相同tile/Problem：route18对12、19对5、20对6、21对7。大部分最优仍是旧路径；route19部分约1%的同tile收益不构成稳定且超过默认最优的证据，route18大多波动内或略慢。当前不推广该epilogue，更不把初筛最快值写入v128正式CSV。

初筛同时发现已有实例还有可验证的分流机会：M2048/N4096/K1536/G4选route6约399.8 TOPS、R1.040；M8192/N1536/K2048/G4选route12约469.5、R1.003。需对相邻M/G做独立正式复测后才能改生产selector，避免单条初筛跨线即推广。

数据见[max_all_v129_packed_screen.csv](max_all_v129_packed_screen.csv)、[max_all_v130_compact_formal.csv](max_all_v130_compact_formal.csv)、[门禁/资源及驻留](max_all_v130_validation_resources.json)。当前正式结果仍是v128的1109 BF16/1123 FP16严格达标。

## v131：相同K64片段顺序，直接读取预排B

v130已证明减少LDS能提高静态驻留但仍未越过小M目标；v126/127把16B加载重排回原8B/lane顺序也有交换代价。新的组合利用INT32整数点积对共同K排列不变：每lane先读16B，在两次MMAC中分别用低8B和高8B；四组lane的K序为q*16+part*8，不必对A和B分别shuffle回q*8。A和B必须同时使用同一个顺序，不能只改一侧。

实验32/33/34/35复用已有prepared-weight格式[K/64,N/16,4,16,16]，每wave连续读取1024B的N16/K64块；A仍读原始MK，不预排动态激活。四种tile为BM16/BN32/BK512、BN64/BK256、BN128/BK256、BN64/BK512。它们仍是Cooperative的load-next/compute-current流程，无输入LDS/CTA barrier，保留原C的LDS epilogue。仅在tuning开关ON可选，M允许masked，不套用WASP整M限制；必须已有合法pack，非pack域在launch前拒绝。

9051/9052/9053把同一K64顺序用于原始B的紧凑LDS、A16B寄存器旁路，去掉旧A shuffle；9054–9057是A/B均原始NT、16B寄存器直取的相同tile对照，用来区分K顺序与B物理预排的组合收益。所有选择仍是实例参数，非case ID分流。CPU已完成454656个贡献位置及144个精确整数点积的预排/排列模型检查；GPU门禁、实际资源和时间未测，不报收益。

同时为v130初筛发现的已有实例机会增加neighbor_formal：N1536/K2048的四档M、四档G，以及N4096/K1536的三档M、三档G，比较默认、route1/5/6/12。正式计时与完整输出验证，用于确定稳定的尺寸分流边界，不直接照抄初筛赢家。

### v131结果

实现2987cd9，API库SHA256为93c0716eace24d0a441a6120896e6c62457653f4ec1b2af34b5a04a1165d22c6。522项full-range INT8/FP64完整输出门禁PASS，72项非pack域在launch前拒绝。所有新kernel无private/spill。输入均寄存器的新实例只保留C的2/4/8KiB LDS；HIP静态上限随几何为8/12/6/6 CTA，不将该数值当动态达到的驻留。

19个shape、760项正式完整输出检查PASS，10 warm/50 repeat/3样本中位数。M1/N4096/K6144/G8，默认2.225 TOPS、R0.973；packed route32为BF16 2.3093、R1.01020，FP16 2.3082、R1.00972，首次在该对照跨过P目标。其余12条小M未达标仍没有全部跨线；K7168部分packed候选还慢于原默认，不做统一替换。M1/N4096/K8192/G64最佳route34约2.393 TOPS、R0.9733，仍不足；短K已达标控制M1/N2048/K512/G64由约2.16到2.45 TOPS，不能把控制样本收益算成剩余目标完成。

同K64原始NT双寄存器与packed对照证明物理B供给的组合影响。例如M1/N4096/K6144/G8的原始B双寄存器约1.79 TOPS，而相同BN32/BK512 packed为2.309；但仍需优化边界/地址等指令，不能宣称所有直接寄存器路径都更好。明细见[max_all_v131_packed_formal.csv](max_all_v131_packed_formal.csv)、[门禁与资源](max_all_v131_validation_resources.json)。v130的78个默认scaled kernel与v127控制逐函数机器码全同，报告在results/max_all_v130_resources/default_scaled_control_equivalence.json，排除可选epilogue参数本身改变对照计算代码。

neighbor_formal独立完成25个原始shape、250项完整输出检查，0数值错误。冻结v130库924dc278...，runner为2987cd9，明确区分编译库与runner来源。M2048/N4096/K1536/G4的route6为399.02 TOPS、R1.0382；M8192/N1536/K2048/G4的route12为469.70、R1.0032，后一条仍接近波动，需要正式全量保持审计。相邻shape确认并非一条规则都选同一实例，例如M2048/N4096/K1536/G8更适合route12，而G16默认已为route12。见[max_all_v131_neighbor_formal.csv](max_all_v131_neighbor_formal.csv)。当前正式全量仍为v128，未用这些小样本替换正式记录。

## v132：FullNK专用实例与WASP CTA访问顺序

小M：在32–35的相同Problem几何增加36–39 FullNK实例。只有host已证明N/BK完整（N%BN=0、K%BK=0）且prepared-weight合法时移除输入N/K判断；M仍逐row masked。检查版保留K尾部支持，FullNK遇部分K在launch前拒绝，不能靠越界读取换性能。新增CPU tuning_domain_test.cpp覆盖整块接受、尾K拒绝、masked M允许、非pack/空/ragged拒绝等19个边界。

大M：不改MMAC/搬运/epilogue，只将现有packed BM256/BN384及BN256的M tile cluster从4分别改为1/2/8/16（48–55）。这里cluster是CTA访问顺序，不是专家组G；原G完全保留。检验相同访存字节下的A/B复用与L2访问顺序，而非继续单独降低LDS。M cluster较大时的小M padding CTA开销计入时间，超出M的CTA在同步前整块退出。先比较代表未达标prefill，不能以部分shape的最佳值宣称全量目标达成。

所有候选仅tuning构建启用，默认selector及v128冻结库不改。待CPU/GPU门禁、元数据和同轮时间验证，暂无收益结论。

### v132结果

实现0505638，API SHA256 a07f29149de43c031dc7a164a3743b454d3acaf649e92a2d7aa34d95b63fd20e。486项full-range INT8/FP64完整输出检查PASS，522项不满足pack/FullNK/WASP域的组合在launch前拒绝，包含masked M及K尾部验证；新kernel无spill。

FullNK正式对照19组、418项完整输出检查PASS，warm10/repeat50/3样本中位数。M1/N4096/K6144/G8的route38为BF16 2.3468 TOPS、R1.0266，FP16约2.346、R1.0264，相比同轮默认约+5.4%。同几何FullNK相对检查版route34约+4.6%，说明边界/编译指令开销会影响纯寄存器方案；并非只要移除LDS即可。M1/N4096/K7168/G8仍约R0.997，M2同族约R1.002、余量很小；其余长K小M仍有未达标。不将初筛逐条最优拼入生产全量CSV。

WASP访问顺序初筛18组、396项输出抽样检查PASS，每组8行，warm3/repeat10/2样本，非正式验收。M-cluster从4改1/2可在部分原默认非最优的尺寸获益，但相同BN384相邻cluster大多约0–1%，不能解决主要大M差距。M4096/N5120/K2304/G48的cluster1约513.70 TOPS、R0.9539，虽比默认+3.7%仍未达标；M8192/N1536/K2048/G4约469.58、R1.0029，接近噪声需独立复测。资源保持：BN384 183VGPR/64KiB/1024threads、BN256 199VGPR/64KiB/768threads，静态最大1CTA。访问顺序没有改变两套核心流水。

归档：[FullNK正式数据](max_all_v132_fullnk_formal.csv)、[WASP顺序初筛](max_all_v132_grid_screen.csv)、[门禁与资源](max_all_v132_validation_resources.json)。默认selector仍v128，正式全量1109 BF16/1123 FP16严格达标不变。

## v133：纯寄存器输入的32位相对字节地址

继续在Cooperative的相同四种FullNK几何做消融，不新增核心流水。40–43对照36–39：仅将A/B相对字节offset算术改为uint32，基指针保留64位；A/B的16B读取、K64共同排列、MMAC、预取顺序及C epilogue不变。已有API检查m*Astride及N*Bstride不超过INT32_MAX；新增实验host检查packed N*K域，FullNK和masked M约束保持。数据位于高虚拟地址仍合法，禁止把基指针截断为32位。

假设：少做64位地址乘加可减少VGPR/地址指令，改善长K、小M供给；反证是ISA无有效变化、寄存器不降或完整时间不改善。先做host边界与full-range数值/guard/scale门禁，再同轮同几何比较；尚未编译、未报收益，默认生产分流不变。

### v133结果

源码b6ed863，API SHA256 95e7473e9ba2e446bf07f9cca6da8d580f00ccca21501b54657ae7bdad5da326。CPU支持域25项PASS，GPU full-range INT8/FP64门禁324项PASS、180项非法组合在launch前拒绝；包括BF16/FP16、无scale INT32、per-token fallback、特殊scale、tail/padded stride及guard。新kernel无spill。19组正式同轮对照342项完整输出检查PASS，warm10/repeat50/3样本中位数，输入[-8,8]，不与full-range门禁混为同一输入域。逐条核对原始样本中位数、2GMNK吞吐及目标R，CSV与jsonl一致。

| BN/BK | 64位offset VGPR | 32位offset VGPR | LDS bytes | 静态最大CTA旧→新 | 完整函数静态指令旧→新 |
|---|---:|---:|---:|---|---|
| 32/512 | 156 | 145 | 2048 | 8→10 | 588→494 |
| 64/256 | 128 | 119 | 4096 | 12→12 | 566→474 |
| 128/256 | 126 | 119 | 8192 | 6→6 | 561→476 |
| 64/512 | 240 | 221 | 4096 | 6→6 | 823→657 |

ISA确认地址进位等指令减少，例如BN32/BK512的v_add_co/v_addc_co各49→7；原MMAC/加载的静态条数保持，但wait/调度也发生变化。上表是整个函数的静态计数，不能当作动态每次循环耗时比例；静态CTA上限不能当实际达到的occupancy。

对同几何36–39，40–43在19组×2输出上的吞吐增益中位数分别仅+0.026%、+0.206%、+0.191%、+0.435%，部分样本回退最多约3%；减少VGPR并不等于稳定整体加速。M32/N2048/K512/G16的新route43约36.66/36.91 TOPS，比同轮默认约+10%，但该条原本已达目标。M1/N4096/K6144/G8的新route41约2.3645/2.3644 TOPS、R1.0344，相对同轮默认约+5.8/+6.0%，这是预排+FullNK+地址的组合相对默认，不能把全部收益归给32位地址。

M1/N4096/K7168/G8约R1.0045/1.0063，M2同族约R1.0077/1.0092，余量仍小；N6144/K7168和N4096/K8192小M族仍未全达标。因此本轮不统一切换到纯寄存器，不把试验赢家拼接进v128全量CSV。现有两套核心流水保持，下一步按各自的复用、物理B访问和资源特点决定是否扩大某个实例的独立验证域。

78个默认scaled kernel与42个无scale/pack kernel逐函数机器码和v132完全相同；public BF16/FP16输入算子未改，默认selector不改。正式全量严格达标数仍1109 BF16/1123 FP16，不宣称1191全部完成。

数据：[v133正式CSV](max_all_v133_address32_formal.csv)、[数值/资源/ISA/代码等价](max_all_v133_validation_resources.json)。操作数路径的选择逻辑另见[OPERAND_DATA_PATH_REVIEW.md](OPERAND_DATA_PATH_REVIEW.md)。

## v134：BN384尾块的无效consumer仅保留barrier

v132访问顺序实验仍有大M差距。源码确认原BN384在主循环为全部12个consumer计算，直到epilogue才屏蔽越出N的32列条带；例如N5120最后一块仅4个consumer有效，另外8个仍搬B、读A LDS并MMAC。新候选只省去完全无效wave的输入读取与MMAC，不改producer，也不省去它们的CTA同步。

pipeline/wasp.hpp新增默认false的MaskInactiveN策略，实验Problem只在packed原方向BM256/BN384启用。56–60分别对应M-cluster1/2/4/8/16，与48/49/12/50/51成对。波内分支统一，inactive consumer初始化acc为0，执行ready一次加num_loop次阶段barrier，随后继续参加所有角色共用的最终LDS sync；禁止提前return。OneBarrier约束保持，旧默认路径不启用。

CPU协议模型检查0–65个K阶段的三种角色barrier epoch、双缓冲读写所有权，以及多种N尾块的32列覆盖；GPU测试需覆盖短/长与奇/偶K阶段、N有/无尾块、多个M CTA/cluster、padded stride、空scale/per-channel/per-token/特殊值、INT32/BF16/FP16输出和guard。模型不能替代编译后GPU门禁，未完成前不报收益。

假设：减少尾块无效wave工作能缓解CU发射/共享LDS端口负担；可能反证为尾块占比太小、分支改变调度/资源或有效consumer依旧是关键路径。先比较相同cluster/tile，再与默认最优比较；无效wave仍占用launch资源，不预报按省掉MMAC比例线性加速。

### v134首轮门禁脚本修正

隔离构建成功，三类旧代码对象（42/78/171 kernels）SHA与v133完全相同，新tail实例182VGPR/64KiB无spill。首轮results/max_all_v134_gates在第一个M1非native fixture的plan_create预期拒绝后，被旧脚本route>=13断言误判：本轮新增了原route12作为对照。进程已结束、没有执行新kernel；不是GPU数值失败或VMFault。修正fixture接受route12的预期拒绝，并给tail suite逐shape增加明确native/packed域的成功/失败断言，避免将应成功的plan_create错误计作正常拒绝。原日志保留，使用新的gates_r1目录验证相同已编译库；仅Python门禁修改，不重建或替换该二进制。

### v134门禁通过与初筛

修正后的r1门禁630项full-range完整输出PASS，360项按显式支持域预期拒绝；全部342个已有kernel在各自代码对象中逐函数机器码与v133相同，新15个kernel均无spill。Mask候选182VGPR/64KiB/1024threads，静态最大仍1CTA，未通过省计算改变静态资源分配。

18组初筛432项抽样输出PASS，每组8行，warm3/repeat10/2样本；不能用作全量性能验收。FP16尾块在相同cluster下中位增益约0.7%–3.1%，BF16混合、部分候选回退。整N控制组也有小幅差异，说明编译调度等影响不能与tail省工作量混为同一归因；cluster8的BF16出现较大回退，需要正式复测核实。不直接推广。

接下来相同18组以完整输出、warm10/repeat50/3样本正式复测默认、原cl1/4/8与Mask cl1/4/8，保留两种输出和原始样本。该runner仅Python新增tail_formal套件，继续用581d61c编译的同一库；不重新编译C++，不把初筛最快样本写入生产CSV。

### v134正式结果及ISA复查

18组正式完整输出252项PASS，warm10/repeat50/3样本中位数；已核对CSV与原始jsonl、中位数及2GMNK吞吐，未用抽样初筛值代替。门禁库SHA256 6773c27d786f76d15b8f5847c863844d18b486a0ccf39fc3303a674f5a004b49，C++源码581d61c、r1门禁runner263fa01、正式runner5dacc5a，明确区分来源。

| 同几何/cluster的屏蔽前后 | BF16有尾块中位增益 | FP16有尾块中位增益 | BF16整块控制中位增益 | FP16整块控制中位增益 |
|---|---:|---:|---:|---:|
| cl1:56对48 | -0.50% | +0.80% | -0.41% | +0.67% |
| cl4:58对12 | +0.54% | +1.82% | -0.39% | +0.25% |
| cl8:59对50 | -1.56% | +3.05% | -4.93% | +0.32% |

每组分别11条有尾块、7条整块；包含K512/K768控制。M8192/N4096/K2048/G8的FP16新route58为505.55 TOPS、R1.0107；M16384/N4096/K4096/G8新route59约624.60、R1.0133；M16384/N4096/K6144/G8约689.27、R1.00193但余量很小。BF16主要差距未解；M4096/N5120/K2304/G48最佳仍原route48约513.73、R0.95395。不能将FP16的结果推广为BF16达标。

BF16 cl8在K512控制224.58→162.49 TOPS（-27.65%），K很长时回退缩小甚至小幅改善，独立正式轮稳定复现；不能归为初筛噪声。cl8 ISA相比旧版多143条静态指令，其中128条v_mov，两个输出格式均出现，来源关联无效acc零值/汇合路径；这是可消除的代码生成开销线索，尚不能证明它独立解释全部BF16回退。新旧主循环MMAC与加载的静态条数未变；完整函数静态长度不能当动态热路径长度。

现有gfx936汇编器实测拒绝v_cvt_pk_bf16_f32及v_cvt_bf16_f32（instruction not supported）；FP16 RTZ转换正向控制接受，但不能用RTZ替换约定RNE。该CPU-only能力探针位于results/max_all_v134_bf16_isa_probe，不改变数值规则，不把硬件BF16转换当可用优化。

归档：[初筛](max_all_v134_tail_screen.csv)、[完整输出正式对照](max_all_v134_tail_formal.csv)、[门禁/资源/342个旧kernel代码等价](max_all_v134_validation_resources.json)。不统一推广，默认生产policy不改，正式全量严格达标数仍1109 BF16/1123 FP16；下一版先消除无效acc汇合，再决定是否保留该尾块策略。

## v135：无效consumer不定义acc，最终同步后退出

针对v134的额外128条move，Masked策略只为有效consumer初始化acc；无效consumer仍完成ready、逐K阶段和所有角色共用的最终LDS sync。CoreKernel与ScaledWaspKernel在run_raw_into返回后、任何epilogue读取前检查MasksInactiveN策略并退出producer/无效consumer。此处之后的native epilogue只有wave内交换/输出，没有CTA barrier，因此不会损坏主循环同步；per-token路径也不会读取未定义acc。原默认false路径保留初始化顺序，数值和接口不变。

仅已有实验56–60启用，生产分流不变；并非靠把无效acc当0或读取未初始化值来省指令。先编译检查move/资源是否实际减少，再执行同一630项full-range门禁与完整输出正式对照，不预报收益。

### v135结果

源码27c6cbf，full-range门禁630项PASS、360项显式域拒绝；18组252项正式完整输出PASS，warm10/repeat50/3样本中位数。逐条核对M/N/K/G/candidate/output、CSV/jsonl、中位数、TOPS和冻结目标R，目标CSV SHA与运行manifest一致。精确库SHA、runner、资源和ISA见max_all_v135_validation_resources.json。所有新kernel无spill，182VGPR/64KiB/1024threads，静态最大仍1CTA；342个旧kernel机器码完全不变。

cl8的额外128条v_mov确实消失：BF16完整函数静态指令10417→10296（原对照10274），FP16 7966→7845（原对照7823）；新增加的仅控制/判定等22条净静态指令，不把静态数量当动态耗时。v134的BF16明显回退不再复现。两轮不是跨版本交错计时，因果归属仍以当前库内相同cluster的独立对照为依据。

| 同几何/cluster屏蔽前后 | BF16尾块中位增益 | FP16尾块中位增益 | BF16整块控制中位增益 | FP16整块控制中位增益 |
|---|---:|---:|---:|---:|
| cl1:56对48 | -0.76% | +0.60% | -0.41% | +0.43% |
| cl4:58对12 | +1.08% | +1.83% | +0.30% | +0.35% |
| cl8:59对50 | +2.68% | +3.08% | -0.17% | +0.29% |

每组11个有尾块、7个整块控制。cl1 BF16并非稳定获益，不能统一替换。M8192/N4096/K4096/G8的cl8约621.49 BF16 TOPS、R1.00826，FP16约624.78、R1.01359；M4096/N5120/K2304/G48约527.22/533.96、R0.9790/0.9915，虽比同轮默认+6.47%/+4.60%仍未严格达标。M8192/N4096/K2048/G8的BF16 R0.99934，M16384/N4096/K6144/G8的BF16 R0.99955，均不可按1算；FP16的R1.00016也没有稳定余量。

归档：[完整输出正式CSV](max_all_v135_tail_formal.csv)、[门禁/代码等价/资源/ISA](max_all_v135_validation_resources.json)。默认policy仍v128，不将局部最优拼进全量结果；正式1191严格达标仍1109 BF16/1123 FP16。下一步对所有满足M>=2048且整M256、N>=1536且有BN384尾块、N%256==0、K为1024到16383且K%128==0的原始shape扩大同轮对照，确认通用尺寸分流和无回退，再决定生产启用。caseID仅识别测试行，不参与运行时dispatch。

### v135尺寸邻域完整覆盖结束

同一v135库、runner47d507c在卡7完成156个原始shape、1872项完整输出检查，0数值错误、0跳过、0不支持。覆盖13个(N,K)族，全部M>=2048且M%256==0；每个shape比较默认、route6、原BN384 cl4/cl8、mask cl4/cl8，BF16/FP16两输出。计时warm3/repeat10/2样本中位数，属于分流校准筛选，不是最终全量验收；输入[-8,8]完整输出，完整范围及特殊scale门禁为前述630项。已逐条校验CSV/jsonl、原始样本中位数和2GMNK吞吐。

选择规则：仅80-CU gfx936、已有合法packed B、M>=2048且整M256、3584<=N<=8192且N%256==0并有BN384尾块、1024<=K<16384且K%128==0；K>=4096或尾块有效N为128时用cl8，其余用cl4。N2048的9条保留原路径，其余147条适用。这个规则没有caseID表、没有目标值runtime查询，INT32无scale的原selector不动。

按该规则在这轮156条内无观测回退，吞吐提升中位数BF16 +1.73%、FP16 +2.25%；严格R>=1由BF16 130→136、FP16 132→140，均没有旧达标项掉线。该筛选只是规则校准证据，不能直接加到旧全量1109/1123上，也不能将小余量跨线视为稳定保证。完整数据见[max_all_v135_tail_neighbors.csv](max_all_v135_tail_neighbors.csv)。

## v136：将已验证的尾块策略接入正式scaled实例

新增WASP Problem/实例MaskedWideM4/M8及稳定route22/23，保持已有route0–12和ABI结构、9个入口不变。prepared B绑定显式包含22/23，I32与scaled private dispatch均支持它们，但普通无scale INT32仍用原selector；只在BF16/FP16输出策略中按上述尺寸域启用。新增Problem直接复用现有WASP pipeline的MasksInactiveN协议，不引入第三套核心流水，不依赖experiments目录。

新增host边界/未校准范围断言、生产API full-range夹具（cl4、N尾块cl8、长K cl8，含padded B/输出scale模式），保留原INT32路径对照。使用TUNING=OFF的独立构建和1191条单轮正式runner，先跑现有完整数值/ABI/lifetime门禁，再全量。全量期间冻结C++及测试源码，尚未构建/运行前不报最终提升。

SingleB草稿仍只在本地隔离目录，未并入v136；将作为后续结构候选，不能覆盖本版新增的生产route22/23和fixture。

### v136正式构建门禁完成

源码06a442c，TUNING=OFF生产库SHA256 e026eeccd82efada138891ffeccdb36bae9523dae570c4add1cee63c46a8d924；benchmark SHA256 e964c6bb62a54ab1a8594ffc5a997aa04e071200fefdbab3c3a31f2d6e0ba0e7。CPU selector边界、34项tuning域与3333项barrier协议模型PASS。GPU full-range INT8/FP64完整输出382项PASS，非法配置7项拒绝，已包含新route22/23的三种生产API夹具和原INT32分流对照；native BLAS scalar reference6项bitwise PASS。epilogue18项、原INT32 ABI边界65项、生命周期32项均PASS，scaled/ABI/lifetime报告均匹配该库SHA；epilogue旧报告不独立嵌入SHA，由相同冻结门禁driver环境执行。

120个原有生产kernel逐函数机器码与v135完全一致；新6个生产入口（2个cluster×INT32/BF16/FP16）与对应已测tuning实例机器码完全相同。新实例182VGPR/64KiB/1024threads、无spill，HIP静态最大1CTA。证明从实验迁移未引入额外设备代码变化，不以该证明代替新分流的全量性能验收。

证据见[max_all_v136_validation_resources.json](max_all_v136_validation_resources.json)。接下来只用该OFF库，在卡7对原始1191条做一轮BF16/FP16正式回归，warm10/repeat50/3样本中位数，完整输出/guard/输入scale抽样不变检查；准备时间单列。全量期间冻结编译及测试源码，不把156条筛选赢家拼接为全量结果。

### v136全量已启动（最终结果待审计）

卡7正式进程PID444065，目录results/max_all_v136_full_1191，使用关闭实验覆盖的e026eecc...库；source revision记录f27aacf，编译实现06a442c，两者差异仅门禁文档。原1191个shape、881 decode/310 prefill，一轮、双输出、warm10/repeat50/3样本中位数，完整输出/guard/输入scale不变检查。启动后已观察真实进程和PASS记录，未以pid文件代替存活证据。

原Excel于本轮重新共享读取核对，SHA256仍为adbb420c4e4d9a3a7a30520c8af55369b94ae79f54ba3b8c0410c1fe2a6478ae，与冻结P列目标一致。当前尚无全量完成结论；结果以该目录done.json/failure.json、原始records.jsonl及完成后的独立审计为准。期间只同步本条文档及Git bundle，不重写冻结C++、测试或runner文件。

### v136全量结束与独立审计

卡7PID444065的1191条单轮正式回归完成。BF16/FP16各1191条完整输出PASS，0失败、0跳过，原始881 decode/310 prefill均保留。独立核对原始shape/E=G、phase、单轮、各3样本中位数、2GMNK吞吐、冻结P目标/R、guard、输入/scale抽样不变、总表及两个单输出表，全部一致。done只在运行结束源码快照与开始一致后产生；另从实现提交06a442c的Git blob精确重建71份编译/测试/runner源码，对9个原有CRLF文件按冻结SHA进行确切EOL还原，其余直接逐byte匹配，没有改变或忽略源码内容。不是复测最快值拼表。

| 输出 | 严格R>=1 | R>=0.99（仅观察） | 未严格达标 | 最小R | prefill严格达标 | decode严格达标 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 1116/1191 | 1130/1191 | 75 | 0.94270 | 249/310 | 867/881 |
| FP16 | 1131/1191 | 1151/1191 | 60 | 0.95278 | 262/310 | 869/881 |

比v128的1109/1123条，全量净增7/8。147条路由实际变化（84条route22、63条route23），两轮吞吐比中位数BF16 +1.70%、FP16 +2.33%；不是交错计时，仍以之前同轮对照作为方法证据。改动行中有6条BF16、7条FP16从<1跨到>=1，没有旧已达标改动行掉线。另2条BF16、1条FP16跨线发生在未改路径，不能归为新分流收益；未改路径ID1131的BF16从R1.00112到0.99857，设备码和选择不变，仍按本轮实际结果记录未达标，不用新测最快值覆盖。147行BF16最小速度比0.99929（约-0.07%），FP16最小1.00791；不把“达标未掉线”误写为每个时延都严格提升。

| shape M/N/K/G | BF16 TOPS | BF16 R | FP16 TOPS | FP16 R |
|---|---:|---:|---:|---:|
| 4096/5120/2304/48 | 527.462 | 0.97944 | 534.193 | 0.99194 |
| 8192/4096/4096/8 | 621.453 | 1.00819 | 624.830 | 1.01367 |
| 16384/4096/6144/8 | 687.581 | 0.99947 | 689.385 | 1.00210 |

第一条原BF16为497.254 TOPS、R0.92335，现已明显靠近目标但仍未达标。所有剩余项的并集75条；BF16为61 prefill+14 decode，FP16为48 prefill+12 decode。N3072/K4096仍有16条未达标，是整N块路径的重要剩余族；小M仍需单独处理。不能把数值全部PASS、R>=0.99或不足1的四舍五入数当作全目标完成。

当前gfx936汇编器正向接受现用v_mmac_i32_16x16x32_i8；对所试v_mmac_i32_16x16x64_i8及v_mmac_i32_32x32x16_i8助记符报invalid instruction，故不把这两个未被工具链接受的候选当作可用加速路径。探针仅CPU汇编，位于results/max_all_v136_mmac_probe，不宣称涵盖全部可能指令。

归档：[全1191条双输出CSV](max_all_v136_1191.csv)、[BF16 CSV](max_all_v136_bf16.csv)、[FP16 CSV](max_all_v136_fp16.csv)、[剩余75条](max_all_v136_remaining.csv)、[全量审计统计](max_all_v136_summary.json)、[路由变化及跨线归因](max_all_v136_policy_changes.json)。原始样本/manifest/报告保留于results/max_all_v136_full_1191。当前生产候选仍为v136，未触及其它16位输入接口；原INT32策略保持。目标尚未完成，下一轮继续验证WASP单B缓冲的驻留/延迟取舍，本地草稿尚未构建或GPU测试。
## v137：WASP单B寄存器缓冲与独立CTA（准备验证）

v136正式全量仍为BF16 1116/1191、FP16 1131/1191严格达到Max ALL P目标；本轮不改变默认选择。此前BM128/BN384虽然把A LDS降到32KiB，896线程、119VGPR组合下HIP最大驻留仍只有1CTA。下一步用同一WASP流水减少B的活跃缓冲，检验资源约束能否改变，不将更少LDS/VGPR等同于更快。

Pipeline末尾增加默认false的SingleB策略：B使用slot0，当前K stage的low/high MMAC全部完成后才发下一stage加载；下一stage消费前沿用VMEM wait。A仍用两个LDS槽，producer、发布/退役barrier及最终LDS同步不变。候选61–64为BM128/BN384、AReadBatch1/2与M-cluster2/4；65–67及已有13为同几何双B缓冲对照。实例编号只用于tuning，不是caseID，生产分流与原INT32路径不改。

预测B单缓冲可减少约16个活跃I32寄存器，可能使每CU最大CTA从1到2；代价是同consumer的B-next/当前MMAC重叠减少，且M128相比M256会增加B读取。必须检查实际VGPR/LDS/spill/HIP occupancy和完整kernel时间。若仍只有1CTA或额外等待超过收益，不推广；转而评估更少consumer wave、每wave更宽N的联动结构，而不继续仅扫描旧参数。

CPU B生命周期模型16900检查PASS（非GPU证明）；GPU将覆盖full-range INT8/FP64完整输出、INT32/BF16/FP16、scale模式/特殊值、K奇偶stage、尾部/stride及guard。先代表prefill初筛，再对有希望的候选独立正式计时；未经正式验证不改生产，不拼接赢家覆盖1191正式CSV。旧生产kernel将做逐函数机器码对照。当前新候选未编译、未运行GPU，尚无收益结论。

### v137本地门禁与执行状态

源码86e9e1b已同步到ck_yyc指定仓库；本地runner语法检查及B生命周期16900项检查PASS。GPU计划为648项full-range完整输出与216项非法域prelaunch拒绝检查，随后查询实际occupancy并做同轮代表shape对照。这些GPU检查尚未执行，不能将计划条数写成PASS。

构建启动与GitHub推送均被自动审批服务的用量限制阻止，命令未执行；未启动编译/GPU进程，也未绕过审批。服务器源码已同步86e9e1b，最后确认GitHub为f8c4570；恢复后先核对远端HEAD/进程再构建，不重复启动。状态记录见max_all_v137_local_preflight.json。正式生产仍为v136，性能达标数未因本地模型检查改变。

### v137 GPU验证及初筛结果：不推广单B缓冲

自动审批服务恢复后，已按原审批流程完成构建及GitHub推送。隔离构建build/max_all_v137_single_b，源码86e9e1b，API SHA256 b35c856ab48c76b7765ea5dbf42b013e781e240d8ddd63cd3385baeea61f4d05。648项full-range INT8/FP64完整输出检查PASS，216项非法域在launch前拒绝。覆盖INT32/BF16/FP16、无scale/标量A加channelB/per-token A/特殊scale、odd/even K阶段、尾部/stride/guard。384个kernel无spill。旧生产126个kernel与v136逐函数机器码完全相同，旧v135实验及生产357个kernel也全部相同。

单B缓冲确实降低资源：BM128/BN384、896线程、LDS32KiB，AReadBatch1为95VGPR，batch2为103VGPR；HIP静态每CU最大CTA均为2。同几何双B的batch1/2为111/119VGPR，静态仍只有1CTA。这是资源可行上限，不是动态achieved occupancy。

卡7独立完成16个代表prefill/回归shape、320项完整输出初筛，0数值错误；3 warm/10 repeat/2样本中位数，保留所有原始样本。单B没有新增填平任何本轮默认未达标项，多数中长K仍慢于默认。例M16384/N4096/K6144/G8，BF16默认687.61 TOPS、单B最佳598.21（约-13.0%）；M8192/N3072/K4096/G6，默认613.59、单B最佳557.19（约-9.2%）。双B batch2同几何对照中，单B的BF16吞吐比中位数约0.990、FP16约0.977，驻留上限翻倍未抵消单wave预取重叠减少及不同tile的额外B请求等成本。时延结果证明该组合没有整体收益，尚未用动态counter拆分各成本，不能把单一原因写成已证。

M2048/N4096/K1536/G4的单B BF16约405.76、默认395.24，但该行已超过目标，且FP16无对应收益，不晋升生产。初筛另观察已有route12对N4608/K5120代表行有收益；尚未独立正式复测及相邻域验证，不改selector、不覆盖正式全量数据。

按用户要求，本轮不再启动新实验，先归档并梳理当前实现。单B实验只保留在tuning开关，未启用生产；1191正式结果仍是v136的1116 BF16/1131 FP16严格达标，75/60条未达标。初筛不是最终验收，不以逐条最快拼接正式CSV。报告见max_all_v137_single_b_screen.csv、max_all_v137_screen_summary.json及max_all_v137_validation_resources.json；总流程见CURRENT_INT8_NT_STATUS_AND_FLOW.md。

### 源码流程补充核对：现有产物的指令与阶段

仅使用v136冻结代码对象反汇编，没有编译、启动或计时新GPU实验。所选scaled Cooperative BM16/BN32/BK512的实际指令包含global_load_dwordx4、ds_write_b128及ds_read2_b64，LDS25344B；WASP BM256/BN384原方向使用buffer_load_dwordx4 ... lds直接装入A的LDS槽、普通buffer load装入B的VGPR、ds_read_b128读取A，函数无显式ds_write，LDS65536B。归档current_flow_instruction_evidence.json保留函数名、代码对象SHA及指令例子。静态函数计数不作为动态每stage计数或耗时比例。

逐条核对正式清单M1/N4096/K7168/G8为route0/legacy8734、M8192/N3072/K4096/G6为route12，补充了CTA/wave任务范围、单/双缓冲层次、K128 low/high时序和stage barrier发布/退役的联合条件。所有容量、逻辑字节和CTA数由实际实例定义推导，并与资源元数据交叉核对；逻辑读字节不当作实测HBM/LDS流量。文档见CURRENT_INT8_NT_STATUS_AND_FLOW.md第5.1、6.3节。生产及1191结果没有改动。

继续按用户要求做结构梳理，未启动新实验。核对consumer N32在accumulator、B地址、wave数量、两个epilogue和尾块判断中的耦合位置，区分接口契约与实现选择；用同一M256/N384/K128有效区域对比当前结构及v137，明确M128重复B请求但未减少A consumer逻辑读取的代价。N64 consumer只作为未实现的任务划分讨论草案，实际资源和性能未知，不将估算写成收益。见总流程文档第9.1、9.2节。

## v138：恢复优化，验证已有实例的分流邻域

在用户明确恢复优化后，先复用已通过v137门禁的tuning库，不重编译C++。新增frontier_v138套件，按尺寸选择N4096/K6144或7168、M<=8的全部原始shape，以及N4608/K5120、M>=256的全部原始shape；包含已达标控制，不以caseID限制运行时适用域。分别与route41和route12做同轮交错对照，两种输出、完整输出验证、warm10/repeat50/3样本中位数，保留全部样本及库/runner身份。

假设：小M的prepared-B寄存器路径以及完整N384块的WASP实例仍有未接入默认策略的收益。若相邻域出现回退或余量不足，不推广；先不改变生产分流，也不替换v136全量CSV。

### v138正式对照结果

54条原始shape、216项完整输出全部PASS，0不支持/失败；已独立核对jsonl、每项3样本中位数和2GMNK吞吐。库仍b35c856ab48c76b7765ea5dbf42b013e781e240d8ddd63cd3385baeea61f4d05，runner提交41b3358。N4608/K5120的6条邻域中，仅M2048/G12默认不是route12；换为route12后BF16 R0.99389→1.03984（+4.62%），FP16 R1.00465→1.04523（+4.04%）。其余5条原本已是route12，同一kernel重复计时的差异不算优化。

48条小M对照中，route41整体BF16速度比中位数0.97467，不能推广到整个长K域。N4096/K6144、G8时M1/2/4/8约+5.6%～6.5%，G16只约+0.6%～1.4%，G32/64回退；K7168仅G8约+1.0%～1.4%，G12及以上回退。N4096/K6144/M1/G8本轮R1.0416，K7168/M1/2/G8为R1.0057/1.0106。小M候选尚需移出实验目录、生产域与full-range门禁，先不接入。v136全量结果仍1116/1191，不能把候选样本拼为新全量成绩。

归档：max_all_v138_frontier_formal.csv、max_all_v138_summary.json；原始样本与manifest位于results/max_all_v138_frontier_formal。

## v139：把完整N块的调度阈值接入scaled生产策略

基于v137初筛和v138正式同轮对照，在已测N4608/K5120、M2048～4096且整M256、prepared B、80CU域，按实际CTA批次数量判断是否从原route5转route12：old_batches>=20，且old/wide>=1.46。只调整BF16/FP16的scaled策略，不按caseID分流；INT32策略不变。原1191清单只有ID992应改变路线，门禁包含无packed、其它CU和不完整M的边界。尚待独立TUNING=OFF构建和GPU验证，当前不计新增达标数。

### v139生产验证完成

实现5c1e496，TUNING=OFF库SHA256 c6126c0b7b223108338304325bd38c00b5dd06bda4e99e76f809d05efe20ec90，benchmark SHA256 6b5eb721c68fd7a883a82a42c27973c97b1c335295d5bcdc34b08778f8055bd6。两个设备代码镜像与v136逐byte一致，126个kernel全部未变、无spill，本版只改变host策略。实际C++策略按完整native域核对1191行，仅ID992的scaled路线5→12；审计脚本首版漏了N/K native限制，已纠正后核对，不把域外函数返回当实际dispatch。

CPU selector边界、39项tuning域、3333项尾部协议PASS；GPU完整范围382项、7项非法域拒绝、实际BLAS标量对照6项bitwise、epilogue18项、原INT32 ABI65项、生命周期32项均PASS。生产库在卡7独立完成N4608/K5120全部6条邻域，两输出12项完整输出检查PASS、0失败，warm10/repeat50/3样本中位数。ID992 M2048/N4608/K5120/G12实际默认route12：BF16 1.795126081ms、645.994274TOPS、R1.0399268；FP16 1.786114693ms、649.253474TOPS、R1.0451735。其余5条仍达到目标。

本版解决1条已知BF16缺口，但未重跑全1191，不把局部数据拼接为新全量CSV；最近正式全量仍v136（1116/1191，75未达标），其中ID992已有本版独立通过证据。小M route41仍未默认启用，待生产化和支持域验证。

归档：max_all_v139_production_formal.csv、max_all_v139_validation_resources.json；服务器原始证据在results/max_all_v139_production_formal、max_all_v139_gates、max_all_v139_resources。

## v140：低group长K的prepared-B协作实例生产化

将已测route41的共同K排列、FullNK和32位相对地址实现迁到pipeline/packed_cooperative.hpp，原实验文件改为兼容include，避免双份主循环。新增scaled-only稳定route24（BM16/BN64/BK256、两wave）：prepared B、80CU、uniform非空、M1～8、N4096、K6144/7168、G8时启用，取v138有正式证据的域，不扩到已观察回退的G12以上。其余WASP/INT32策略不变。小M不满足WASP native域，所以plan在独立条件下选择该实例；必须同时绑定prepared B，并保持原输出/scale坐标。

新增host域检查和生产API full-range/FP64夹具，覆盖M1/M2/M8、padded输入输出、空scale/标量/channel/per-token/NaN/Inf；INT32须仍走原8734。先独立OFF构建及门禁，再对48条长K邻域正式验证，当前不报新增达标数。

### v140编译与代码等价

OFF构建成功，库SHA256 1e54b515e42f34984bdb5713d85bf9917d9307efac6026ec12edae8a3235126e，benchmark a1ad6e7cd0c649ea18edbc548b41340d6603a0ad19cf7858458475b0164417b3。128个kernel无spill；126个旧生产kernel逐函数机器码与v139完全相同，两个新scaled kernel与此前已验证route41逐函数完全相同。新BF16实例119VGPR、4096B LDS（仅输出交换），主循环输入不经过LDS。实际CPU分流审计1191行，8条小M切route24，加上v139的ID992共9条相对v136改变。

全量runner增加显式门禁版本选择（默认仍v136），核对本次实际库SHA后才可运行；route24报告为cooperative，避免仅用route非零误标WASP。新门禁和48条邻域正式性能验证完成后，使用冻结库进行新1191全量回归，不拼接局部结果。

### v140门禁与48条生产邻域结果

完整范围INT8/FP64检查420项PASS，非法域7项拒绝；实际BLAS标量对照6项bitwise、epilogue18项、原INT32 ABI65项、生命周期32项均PASS。新增route24覆盖完整输出的NULL/标量/channel/per-token和NaN/Inf、padded stride，原INT32仍8734。

卡7的48条原始长K邻域、96项生产输出全部PASS，0失败/跳过，warm10/repeat50/3样本中位数已独立核对；只有G8的8条切route24，其余40条维持route0。新路线8条的两个输出均R>=1。ID1017 M1/N4096/K6144/G8：BF16 0.169539183ms、2.374986TOPS、R1.038957；ID1087 M1/N4096/K7168/G8：0.198380753ms、2.367982TOPS、R1.007275；ID1088 M2同族：0.199593559ms、4.707186TOPS、R1.010333。FP16对应R1.039212/1.006074/1.012329。小余量仍以全量独立回归确认，不用候选赢家覆盖v136 CSV。

源码归一后仍是Cooperative/WASP两家族，scaled实例从41增为42（Cooperative23、WASP19，不按BF16/FP16重复计数）；输出类型分别实例化后新增两个设备kernel，生产总128。正式全1191回归将使用新冻结OFF库和v140门禁版本，保持一轮正式计时及全部原始shape。

归档：max_all_v140_production_formal.csv、max_all_v140_validation_resources.json；原始证据results/max_all_v140_production_formal/max_all_v140_gates/max_all_v140_resources。
