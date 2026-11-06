# 独立 decode 流水原型验证（P6h，实验用，不合入）

本阶段没有形成可以替代现有最优方案的新流水。新实例6000默认关闭，既有正式选择不变。P6g的1193条复现目标继续暂停；本轮没有运行1178/1193全量。

用户追问A放LDS、B放寄存器后，补做了这一方向的可运行原型。此前只实现并测试A放寄存器、B放LDS，未充分比较两种存储分工；不能把此前选择当作已验证的最优架构。

## 当前代码与实验范围

当前源码保留最后一个A-LDS/B-register研究原型：四个wave沿N各负责32列，每个wave覆盖完整M16/32/64输出、遍历全部K；K32两级预取；A进入各wave私有LDS，B由LLVM可识别的raw-buffer指令直接进入VGPR，再组织成MMAC操作数。FP32累加、BF16 RNE写回，带M行谓词，无跨wave归约、无CTA barrier。

这是独立pipeline、provider、CMake archive及decode family，不实例化V7 pipeline或其epilogue。当前三种M配置的gfx936资源为VGPR95/118/107、SGPR51/53/57、LDS8/16/32KiB，scratch和spill均为0。其他中间版本源码与资源证据分别归档。

独立开关`CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_WAVE=1`，同名DISABLE优先；不依赖V7使能。实验能力条件为gfx936、BF16 NN、common非ragged非empty、1<=M<=64、N%128=0、K32..4096且K%32=0、G<=1024，以及selector地址与grid限制。M128/256、prefill、N尾块等走既有回退。只验证了正式benchmark的packed输入；任意自定义stride等公开入口的完整生产契约尚未验收，不可推广启用。公开v1字段布局、语义未改，仅追加ID/family。

未发布实例6000的canonical suffix仍是首版遗留的`decode.nn.wave_private.16x128x16.s2`；它不是当前几何依据。实际配置由模板、launch参数与ISA/profile记录确定。后续正式集成前必须改为通用名称并重建、复验；此阶段不将该原型作为可推广产品。

22条代表shape固定为76/633/681/620/621/358/772和新增1179..1193，14条活动、8条回退。另有19个合成边界/压力用例，覆盖最小K、奇数K32阶段、M31/33/49、G1024、大N及最大K。全部新增15条已纳入。预先声明的100条属性扩围未执行，因为代表域没有足够收益；没有按赢家ID扩围或改选择。

## 各次原型结果

三轮均使用物理GPU4、正式device_args benchmark、默认selector、warmup10/samples30、单launch、include-args-build0；每shape先双方全输出BLAS比对与首尾真实guard，再相邻baseline→candidate三轮，取三轮median的中位数，所有回退/负结果保留。

|版本|设计|活动shape改善>2%|回退>2%|活动耗时变化中位数|
|---|---|---:|---:|---:|
|trial01|4个N-wave，M16，B私有LDS两级|0|14|+26.61%|
|trial02|同几何，A structured/cache-swizzle|0|14|+16.50%|
|trial03|同几何，8级预取|8|6|-3.92%|
|trial04e|单wave全N128，M自适应，显式等待依赖|1|9|+3.42%|
|trial06b|A私有LDS，B寄存器，K32两级|0|14|+18.70%|

这些是不同二进制下各自完整系列，不跨系列取最快值。第一批短K局部收益没有扩成稳定属性域，不能称新流水已成功替代旧方案。最后一个A-LDS原型14条活动均回退，中位+18.70%；其中12条三轮均回退>2%，9条至少一轮CV>5%。

最后版本部分结果（ms；CK两臂均GPU4；Native为来源保留的冻结参考）：

|ID|M/N/K/G|现有方案|A-LDS/B-reg|变化|冻结Native|
|---:|---|---:|---:|---:|---:|
|76|1/2048/768/4|0.021760|0.025760|+18.38%|0.017048|
|633|4/1024/2048/8|0.036880|0.061520|+66.81%|0.032816|
|620|32/1024/2048/32|0.140722|0.200561|+42.52%|0.122144|
|621|64/1024/2048/32|0.146082|0.203443|+39.27%|0.126824|
|772|64/6144/3584/56|1.985523|2.863365|+44.21%|2.035727|
|1179|1/256/512/64|0.031840|0.033440|+5.03%|0.031936|
|1184|32/256/512/64|0.034640|0.037041|+6.93%|0.033504|
|1185|64/256/512/64|0.036160|0.043040|+19.03%|0.031640|

冻结Native旧1178来自GPU0/repeat20，新增15来自GPU4/repeat20，与本轮CK单launch口径不同；不据此证明同口径超越或落后Native多少。完整CSV记录来源、三轮数据、CV、实际实例、epoch、二进制SHA与原日志键。

## 已验证与尚未验证

- 初版四wave/M16切分在M64重复读B：ID772 profiler HBM约9.2GB，对照约2.5GB。单wave或完整M所有权将HBM降回约2.5GB，但本身没有保证时间改善；wave数、寄存器占用和访问组织仍有代价。
- A structured/cache-swizzle消除了ID633原型引入的tag-conflict计数，但该原型仍慢；不能把它当作现有CK与BLAS差距的全部根因。
- 最后A-LDS/B-register，ID633 HBM34.23→33.69MB接近，而VMEM/wave207.5→1089、LDS/wave420→137；ID772 HBM2527.29→2525.72MB，而VMEM/wave462→2244。减少LDS请求没有抵消当前B标量加载及调度代价。LDS冲突计数比例也升高，不能把所有时间差归给单项。
- 当前B加载需要较多ushort请求/打包，LLVM还加入保守vmcnt0。它只验证这个实现，不排除更好的B合并加载/寄存器重排、A跨wave共享调度，或不同K深度；不把“首版慢”说成“A放LDS不可行”。
- 对16x128输出/K32一级，单份共享A仅1KiB、B为8KiB；四份wave私有A为4KiB。M64时四份私有A为16KiB，反而大于B8KiB。比较须统一缓冲K长度并计入复制，不能只比较stage数或A/B矩阵名。

## 正确性、环境与失败保留

最后版本63个精度作业、410次全输出、3,417,443,328个比较元素，max_abs=0，CK/BLAS各32个guard检查通过。22组完整三轮，无当前二进制不完整组。官方selector/query/smoke及7个字节边界回归通过。CPU新选择64,800、默认关闭旧选择241,920、历史两表2386行可达性检查通过。原18个V7 shader字节不变。

本轮997条raw作业均物理GPU4、epoch0，接受作业门禁全部健康，无MKFD或hymgr服务转变，无恢复、reset、服务重启、换卡或终止其他会话进程。统一锁串行、每命令fresh hy-smi/HCU0/主机内存>=64GiB；原P6g期限未重置、Goal仍暂停。

保留4条失败：query首轮继承V7使能导致预期配置错误；profiler带尖括号过滤器未捕获；单wave原型MMAC被LLVM移到LDS等待之前；第一版A-LDS原型opaque asm VMEM结果在等待前被复制。后两项均是正确性失败，未用其性能作采用结论，修复后更换二进制/新键重跑完整组。另有编译错误、资源spill拒绝、gfx936 permlane16编译探针不支持，均仅CPU证据，无GPU性能外推。

最后binary SHA256: `276fdae07237401fdfb56976c7c6c11faa61418deaf8097e4839607da1fbfc99`。两份shape SHA保持`5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863`，用户readfirstlane diff MD5保持`4c7f49a5c7592767d53195a713db6b8a`且未暂存。暂停P6g的`tools/grouped_gemm_reproduce`保留，不纳入本提交。

正式benchmark沿用暂停任务中已准备的默认关闭redzone与非有限值拒绝功能，修复其float-equal构建报错；只在精度作业启用，计时显式关闭。此复用不等于恢复P6g复现。

## 现场命令与证据

现场 `/workspace/v7_coverage/p6h_decode_arch/`。下列入口自身持锁、清理实验开关、执行健康门禁和实例核对：

```bash
python3 /workspace/v7_coverage/p6h_decode_arch/campaign.py sanity
python3 /workspace/v7_coverage/p6h_decode_arch/campaign.py boundary_stress
python3 /workspace/v7_coverage/p6h_decode_arch/campaign.py row_geometry_stress
python3 /workspace/v7_coverage/p6h_decode_arch/campaign.py configuration_min_stage
python3 /workspace/v7_coverage/p6h_decode_arch/campaign.py representatives
python3 /workspace/v7_coverage/p6h_decode_arch/report.py
```

已执行输出包含`SUITE_PASS representatives 22`、`DECODE_REPORT_AUDIT_PASS`、`OFFICIAL_REGRESSIONS_PASS`、`FOCUSED_PROFILES_PASS`、`OLD_SHADERS_PRESERVED_PASS 18`。ISA检查使用`isa_audit.py <benchmark> trial06b`，三个kernel无scratch/spill/barrier。

归档包含各版本源快照、结果与负结果、命令/环境/门禁/源码与二进制SHA、精度/回归/计时/失败raw日志无损压缩。二进制、vendor ISA/CO、巨型perf留现场，不入Git。后续阶段应先改善B寄存器加载与重排，再决定A共享策略；不在当前原型上启用整域，不恢复全量复现。
