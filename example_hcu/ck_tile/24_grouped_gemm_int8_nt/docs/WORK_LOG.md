# INT8 NT 开发记录

## 2026-10-26 范围冻结

用户确认本轮沿用 INT8×INT8→INT32、无 scale；允许读取指定远端仓库及保存必要本地分析副本。卡 7 用于验证。原始 revision 9acdc34e1f733f3c5dbfc85b52d6273409ec2ce8，初始工作树干净。

需求附录 A 共1178条，decode 872，prefill 306，26组 K/N。逐行核对原 CSV 后，旧 INT8 NT 达到 BLAS 50% 的记录为263条，prefill为0。分 M 汇总和文档 SHA 见本目录已有记录。

## 迭代 01 至 03 通用 V4 对照

比较了直接按 group 定位的三维 grid、persistent 调度对照，以及固定热循环/尾部的非 persistent 实例。增加 tile 并扩大 K 向量未获得有效收益，抽测全输出正确，但大 tile 存在明显性能退化。

编译产物资源元数据显示：32×128、128×128、128×256 候选 private segment 分别约332、1512、1172 bytes/thread，VGPR达到256。该信息是寄存器/编译结构需要改变的证据，不能直接断言全部耗时由 spill 导致。旧产物和日志保留在 build/iteration_01 至 iteration_03。

## 迭代 04 显式 packed-INT8 pipeline

小 M 直接 MMAC：K32片段以8字节/线程加载，使用CK库已有MMAC attribute，无LDS。

分块流水：16字节全局加载，K64、LDS行距BK+16，下一tile保持packed VGPR；当前LDS计算完成并同步后才允许覆盖；INT32输出通过分条LDS重排后向量写回。没有更改量化或精度语义。

首批 M/N/K/E=2048/2048/512/64 的128×128候选约1.999ms、137.49 TOPS，旧CK约4.308ms；该数字为开发probe。128×256候选在同一shape约2.93ms。

正式 CMake 开发库已完成6条公开API pilot：ID 1、6、9、35、590、598。默认select/query/run实际选核，两库全输出参考和guard均通过，R分别约1.292、1.639、0.975、0.620、1.885、1.923。这仅证明该6条通过，不代表1178条已经达标。

大K难例ID1158、1177：128tile仅约180 TOPS，距离BLAS的约576/601 TOPS还有差距。

## 迭代 05 资源与大 M 复用

将C暂存改成16行，避免输出暂存刚超过32KiB的LDS驻留台阶；比较256×256、256×128和128×256不同wave划分。ID1158的最佳已测候选约260.66 TOPS；ID1177约281.77 TOPS，仍不足BLAS的50%。抽测全输出和guard通过。

## 迭代 06 CTA 遍历顺序

比较N-first与M-first输出tile遍历，以及K32、M512候选。假设是大专家权重按整N扫描可能影响跨M tile的缓存复用；需要以受控计时验证，不把这一解释当作已确认瓶颈。该轮仍在验证。

## 迭代06时尚未完成的验收（下文更新状态）

- 所有1178条的独立全输出、guard、输入不变和性能记录。
- 公开接口的ragged、empty、非默认stream、stride/tail与错误参数回归。
- 统一C++ benchmark扩展的构建和运行。
- 稳定的最终分流规则、每个候选的资源数据、剩余低于50%的case清单。
- 最终维护源码整理、构建清单与Git差异审查；尚未提交或推送。

## 迭代07至12：加载、资源和参数缓存

比较宽LDS读取、K128、双LDS、去边界检查和peeled循环。双LDS和K128在当前tile上增大资源，没有获得收益；去掉热循环的末尾条件有小幅收益。资源/ISA记录保存在build/iteration_11*：生成指令可见K循环内重复读取参数；将参数复制为局部值后，256×256候选在ID1158/1177达到约327/320 TOPS，全输出和guard通过。

这些数字是开发probe，不是默认selector的正式结果；不把private segment或计时单独解释成已确认的硬件瓶颈。

## 迭代13：公开集成和独立正确性

代码拆为problem/policy/pipeline/epilogue/kernel，移除活跃路径中无收益的实验分支；历史源码保存在experiments和build/iteration_*。公开enum兼容扩展，结构体大小保持problem112、selection208、launch64、candidate_info224字节，C11对照通过。原80条selector回归通过；新用例覆盖aligned/ragged/empty和容量摘要，不允许不满足条件的输入进入无mask路径。

统一C++ benchmark入口已构建和运行，通过CPU精确抽样及完整CK/BLAS输出/guard比较。完整INT8范围和FP64参考的边界回归通过，包括非默认stream和正stride/padding。

首版公开分流96条代表shape全部数值通过，84条达到历史BLAS的50%。其余12条集中在M8192/16384，不能由已通过的中小M结果推断全量达标。

## 迭代14至16：大M调度

比较整个M维连续、M tile分组4/8/16以及N方向优先的遍历。ID902从约128提升到302 TOPS，ID1038从约169提升到305 TOPS；所有被比较候选全输出和guard通过。小K的大M使用128tile、M分组8；大K的大M使用256tile、N优先。分流依据形状与对齐条件，不按模型或phase。

独立CMake的代码生成参数与完整CK grouped-GEMM目标对齐。正式发布候选均无private/scratch分配；元数据见docs/resources_v16.json（该版本13个内核，非后续新增M16候选）。

全部1178条筛选数值通过，1173条达到历史BLAS的50%。低于阈值的ID为1021、1083、1091、1100、1154，均为M16、大K、较少专家的decode。数据位于results/full_screen_01。该筛选的BLAS为历史时间，不能当作同轮验收。

## 迭代17至18：M16专用tile与正式冻结

比较16×128、16×256和两/四wave布局。16×256候选在五条低于阈值的shape上约0.246/0.821/0.271/0.460/0.598ms，旧direct候选约0.624/1.845/0.681/0.981/1.400ms。完整输出和guard通过，均超过各自历史BLAS的50%。该候选只在M16、common无空/不等长标志、K>=4096且K64/N256对齐时启用。

同语义的INT8输出不变，未增加scale、dtype转换或中间输出。统一benchmark和公开ABI的新候选已构建。最终活跃源码按仓库clang-format整理；预编译候选与默认selector一一对应。

正式同轮一轮对比已在results/final_01启动：每样本warmup10/repeat50，三个event样本取中位数，沿用逐shape tuning最快的有效BLAS算法并核对库/算法身份。补测ID1110的有效算法从full_01_supplement_01合并，不能跳过。每条CSV明确INT32输出、无scale、NT和phase。

正式结果与审计以results/final_01/REPORT.md、audit.json、results.csv为准；运行期间不得将筛选或probe数据写作正式达标结论。尚未向GitHub推送代码。

## 最终交付核对：2026-10-27（Asia/Shanghai）

正式一轮完整1178条全部数值PASS、全部R>=0.5，无OOM、VMFault、显存跳过或缺失。最小R=0.508320690，中位R=1.020894423；prefill中位R=0.611534413，decode中位R=1.144740685。独立本地原文解析再次逐条核对ID、M/N/K/E及phase，全部一致。

统一C++入口全1178条数值回归完成，1178 PASS、0失败。它在正式计时后顺序运行，CPU精确抽样+全量CK/BLAS输出/guard验证；time=0证据不替代正式性能。原80条selector、新20条INT8 selector、13组FP64边界、C11 ABI及615行原命令构造回归通过。正式与统一入口的CK gfx936机器码bundle SHA一致；最终14个内核均无private/scratch分配。

结果、构建命令、源码检查点和库/算法身份保存在results/final_01，统一C++原始证据保存在results/unified_validation_1178。CODE_REVIEW.md、PIPELINES.md及FINAL_REPORT.md汇总最终实现与测试。未推送GitHub；原有baseline结果未覆盖。

## 2026-10-28：明确 shape 选核与 instance ID

检查交付版本后确认：INT8 selector 依据实际 M/N/K、架构/布局、对齐及长度属性选择预编译实例，未接受测试 case ID。G 参与合法性、专家 grid 及容量摘要，当前没有独立的 G 性能阈值。instances.cpp 中 case 6004 等为内核 ABI 编号。

本次将 launch switch 的裸数字改成具名 GroupedGemmInstanceId，将参数 id 改名为 instance_id，并补充调用链说明。新增 CPU shape 回归核对原正式 1178 行实例选择及重新编号/乱序不变性。核函数、模板参数和 shape 分流阈值均不变；验证证据见 docs/shape_selection_audit_20261028.json。

验证完成：1178/1178 条 M/N/K/G 选择与正式记录一致；重新编号并反转顺序后一致；20 条原 selector 检查通过。全新构建 14 个内核通过，gfx936 bundle SHA256 与正式测量冻结库完全一致（d381da89b4c098b848c8f265db775603d7914aa0db59e6081d84c037e9c664a2）。本次未执行 GPU 性能复测。

## 2026-10-28 iteration 19

Frozen spreadsheet targets in targets_int8_nt.csv. Two small-M masked packed-LDS candidates improve five screened shapes by 1.95-2.16x; all full outputs/guards pass, all still below targets. Experimental only. Details: iteration_19_small_m.md.

## 2026-10-28 iteration 20

CTA-local K partition (2/4/8 waves), all 25 screened outputs/guards pass but slower than v19. Not promoted. Exact BLAS kernel ISA located via /opt/dtk/aillvm/bin/llvm-objdump. Static summary and iteration_20_local_k.md record evidence and limitations.

## 2026-10-28 iteration 21

Combined contiguous packed transfer with wave-local K partition and in-kernel INT32 reduction. All 25 screened outputs/guards pass; shape-dependent improvement over v19, still below spreadsheet targets. No public selector change. See iteration_21_packed_local_k.md.

## 2026-10-28 iteration 22

BK128/256 small-M packed-LDS experiments; all 35 outputs/guards pass. BK256 improves screened performance; M16/N7168/K3072/G12 is faster than spreadsheet target in screening. Expand coverage before production promotion. See iteration_22_deep_k.md.

## Iteration 22 expanded small-M screening

560 shapes x 8 candidates, all 4480 full output/guard/input-sample checks pass. Best-per-shape oracle >= target: 546/560; median ratio 1.352390; min ratio 0.958686. This is candidate screening, not production selector acceptance. Shape rules and independent validation remain required. Summary and best rows saved in iteration_22_full_screen_summary.json and iteration_22_small_m_best_screen.csv.

## Iteration 23 public dispatch

Integrated three small-M K128/K256 instances with K/G shape rules. 21 INT8 and 80 original host selector checks pass; 18 full-range boundary tests pass. Public 560-shape validation launched, pending results. See iteration_23_public_dispatch.md.

Iteration 23 completed: public API 560/560 numerical PASS; 545/560 meet fixed workbook throughput. Shape-only selector replay covers all 1178 rows; larger-M expected instances unchanged. Full-goal work continues.

## Iteration 24 large-M existing candidates

Four representative large-M/large-K shapes x four existing instances; 16 full output/guard checks pass. Best ~300-325 TOPS, still below target; N-first helps M16384 but dispatch-only changes are insufficient. Next investigate operand supply/read scheduling. See iteration_24_large_tile.md.

## Iteration 25 layout experiments

128 small-M and 24 large-M full output/guard checks pass. Narrow BN32/BK512 meets all 15 remaining small-M targets in screening. Wide paired LDS reads improve large-M only slightly, insufficient for target. BLAS large-M direct-to-LDS instructions confirmed. See iteration_25_layout.md; production path unchanged this iteration.

## Iteration 26 small-G long-K dispatch

Added shape-only 6505 route, K512/BN32. 80 original plus 24 INT8 host checks and 21 full-range boundaries pass. Public 560-shape target verification pending. See iteration_26_small_g_dispatch.md.

## Iteration 27 native D2L primitive

Linear and XOR layouts round-trip exactly, OOB loads zero-fill preinitialized LDS, all guards pass. Copy probe only, no GEMM throughput claim. See iteration_27_direct_lds_probe.md.

Iteration 26 completed: public API 560/560 numerical PASS; 560/560 meet fixed workbook throughput. Shape-only selector replay covers all 1178 rows; larger-M expected instances unchanged. Full-goal work continues.

Iteration 26 evidence filenames corrected: restored iteration-23 audit/CSV/dispatch report from their original checkpoint; iteration-26 evidence is now stored separately. No GPU retesting or metric changes.

## Iteration 28 native D2L GEMM

20 full-range FP64 oracle checks and 28 performance-output checks pass. Ping-pong reaches ~328-367 TOPS on four large-M cases, still below targets; producer specialization does not improve this variant. Retained experimental, no default change. See iteration_28_direct_lds.md.

## Iteration 29 LDS K-vector mapping

28 full-range oracle combinations and 32 performance-output checks pass. Shift1 modestly improves some cases; shift2 regresses. No default change. See iteration_29_lds_mapping.md.

## Iteration 30 wave geometry and tile grouping

40 full-range oracle checks and 48 performance-output checks pass. Best352-395 TOPS on representative large-M shapes, below targets. No default change. See iteration_30_geometry.md.

## Iteration 31 medium-M packed deep-K

44 full-range oracle checks and 1568 screening records pass. Best-candidate coverage192/196; no production change. Residual small-G/long-K requires further work. See iteration_31_medium_m.md.

## Iteration 33 medium-M packed deep-K

36 full-range oracle checks and 1648 screening records pass. No production change; see audit for best-candidate coverage; no production change. Residual small-G/long-K requires further work. See iteration_33_medium_m.md.

## Iteration 32 D2L depth and compiled resources

32 full-range checks and 40 performance-output checks pass. K32 resource savings do not improve large-M throughput. Actual metadata confirms v30 four-wave candidate private allocation. No default change. See iteration_32_depth_resources.md.

## Iteration 34 medium-M packed deep-K

40 full-range oracle checks and 1556 screening records pass. All original packed candidates remain best; no production change. Residual small-G/long-K requires further work. See iteration_34_medium_m.md.

## Iteration 35 medium-M packed deep-K

32 full-range oracle checks and 1280 screening records pass. Limited wins; no default change. Residual small-G/long-K requires further work. See iteration_35_medium_m.md.

## Iteration 37 medium-M default dispatch

Five packed deep-K instances registered using M/K/G thresholds. Host80+30, CPU1178 replay and28 full-range public boundaries PASS. Full196 default-entry performance run follows. No all1178 throughput claim. See iteration_37_medium_m_dispatch.md.

## Iteration 36 native D2L PMC

Captured one real GPU7 dispatch with installed hipprof PMC; full output check passes. Raw counters and interpretation limits recorded. v30/v32 experimental ELF extraction was corrected and exact original hashes restored. See iteration_36_pmc.md.

Iteration37 public run completed:196/196 numerical PASS, 192/196 meet fixed target. Four misses are retained; remaining optimization continues. All metrics use default selection and original dimensions.

## Iteration 38 M128 coverage

All78 shapes numerical PASS across620 development records; best-candidate target coverage76/78. No default change; next isolate narrow BN for small-G residual.

## Iteration 40 M128 narrow-N

24 full-range checks and620 full-output screening records pass. Best-candidate coverage77/78; one small-G residual remains. No default change. See iteration_40_m128.md.

## Iteration39 aligned/native-only loop

40 full-range checks,5 >2GiB stride-span fallback checks and28 performance-output records pass. Hoisting improves4-6% on four representatives, still below targets. No default change. See iteration_39_aligned.md.

## Iteration41 frozen BLAS replay

GPU7 algorithm6950 reproduces603.452TOPS without profiler; full output/input/guard checks pass. Actual launch is768 threads and64KiB LDS; resource and raw PMC comparison saved. Fixed Excel target remains unchanged.

## Iteration42 no-LDS output

40 full-range checks,10 odd-output-stride/large-input-span checks and32 performance-output records pass. Scalar output regresses; wave-transposed vector output improves modestly on large M. No default change. See iteration_42_direct_output.md.

## Iteration43 packed input/direct output screen

128 full-range combinations and4196 full-output records across834 shapes pass. Best-candidate throughput meets830/834 targets; four residuals remain. Small-M long-K narrow-N K512 candidate improves margin. No default change. See iteration_43_direct_epi.md.

## Iteration44 matched PMC

All three paths pass full input/output/guard checks; corrected name filter captures9 dispatches. No-LDS output removes reported bank conflicts but only modestly improves time; do not treat that metric as the dominant bottleneck. See iteration_44_pmc_compare.md.

## Iteration45 K32 streaming ring

80 full-range checks and48 performance-output records pass. No spills, but K32 ring-buffer variants regress. Retained experimental; next change physical copy tile and role lifetimes. See iteration_45_streaming.md.

## Iteration46 dense K128

72 full-range combinations and44 performance-output records pass. All-wave variants reach about381-419TOPS; mixed paired-producer variants spill6-7 VGPRs. Historical masked offset-proof gap documented; no promotion. Next split role loops/lifetimes and include padded K in the stride guard.

## Iteration48 split role loops

54 full-range checks (48 boundary combinations +6 repeated smoke checks),36 performance records pass. Separate wave-role lifetimes remove spills:233-234 VGPRs,zero scratch. Representative throughput remains below the frozen BLAS targets; no default promotion. See iteration_48_split_roles.md.

## Iteration47 small-M public integration

Reusable no-LDS epilogue; uniform M<=16,K>=4096 selects8734 for every G. Host80+33, CPU1178 relabel/reverse,28 full-range public boundaries PASS. All560 small-M public throughput checks follow. See iteration_47_small_m_dispatch.md.

Iteration47 public run completed:560/560 numerical PASS, 560/560 meet fixed target; minimum R=1.007298, median=1.366718. Default selection, original dimensions,28 independent full-range boundary profiles PASS. Remaining618 shapes not accepted by this run.

## Iteration49 large-screen validation memory

Original screen stopped at132/344 shapes (792 valid records), active shape464. Full bool-to-int64 reduction temporary requested42GiB. Preserve prefix and resume exact full-output comparisons in256-row chunks; no kernel/workload/timing change. See iteration_49_screening_oom.md.

## Iteration50 A double buffer/B bypass

66 full-range checks and32 perf-output records PASS. New structure uses A-only two32KiB LDS slots and direct B registers.256 VGPRs,24-28 spills; own ISA shows simultaneous loading of both A halves at final compute. Keep experimental; test stronger compiler lifetime boundaries next. See iteration_50_bypass_b.md.

Iteration49 screen completed:344/344 original large-M shapes,2064 numerical/guard checks PASS. Oracle12/344 fixed targets; no public promotion. Full CSVs and audit preserve every tested candidate and phase.

## Iteration51 compiler lifetime controls

66 full-range and32 performance-output checks PASS. Memory-only clobbers leave24-28 spills; volatile vector LDS loads regress. Continue with accumulator-tied MMAC ordering, no public default change. See iteration_51_lifetime.md.

## Iteration52 ordered MMAC policies

Native instruction and intrinsic-plus-tied-fragment compiler boundaries compile to244 VGPRs,zero scratch/spills.66 full-range and32 performance-output checks PASS. Representative best417-484TOPS; fixed targets still missed, no default promotion. Next matched operand-path counters. See iteration_52_ordered_mmac.md.

## Iteration53 matched operand counters

All3 full-output/input/guard checks PASS,9 target dispatches. Direct B reduces aggregate LDS-wait but has more VALU work and unchanged total LDS instructions; not a stall-time percentage. Local BLAS adapter confirms swapped operands/C-transpose mapping. Next test equivalent internal transposed task and contiguous scalar C stores. See iteration_53_operand_counters.md.

## Iteration54 internal transposed mapping

External inputs/output unchanged; scalar contiguous stores from internal B*A^T.66 full-range and32 performance-output checks PASS;243 VGPRs,zero scratch/spills. Gains vary with shape; frozen targets not reached. No public default promotion. See iteration_54_transposed_mapping.md.

## Iteration55 bulk raw copy

88 full-range and40 performance-output checks PASS,243-244 VGPRs,no scratch/spills. Hoisted offsets/bulk m0/scalar descriptor advancement give only small shape-dependent gains. Both internal orientations retained as experiments. No default change. See iteration_55_hoisted_copy.md.

## Iteration56 fresh same-input comparison

12/12 full-output/input/guard checks PASS. Original BLAS algorithm6950 gives601.7TOPS; CK candidates481-486TOPS. Alternating order,5warm/20repeat,3samples, same allocations. Frozen target remains unchanged; gap is reproducible. See iteration_56_fresh_matched.md.

## Iteration57 structured B/cache-swizzle

120 full-range checks and40 performance-output checks PASS. No spills. Cache eligibility depends on physical stride. Representative gains remain below fixed targets; no default promotion. See iteration_57_structured_b.md.

## Current public-v47 full1178 snapshot

{"numeric_pass": 1178, "ge1": 832, "ge05": 1164, "min_R": 0.45631943431758887, "median_R": 1.198147838917571, "library_sha256": "865755915e5f4b9045d40689311ca531f7980b7531b9ac0b4cf22748a51e9bb5", "phase": {"decode": {"total": 872, "ge1": 825, "median_R": 1.329005798024912}, "prefill": {"total": 306, "ge1": 7, "median_R": 0.607738874675903}}}
See current_public_v47_1178.csv; all rows share one library identity,560 reused and618 newly measured.

## Iterations58-63 WASP correctness isolation

Native fragment/full port failed. Static A/B reconstruction passes; ordered MMAC alone fails. Short A-fragment lifetime passes eight small controlled diagnostics. Full kernel/performance remain unverified. See iteration_58_63_diagnostics.md and exact JSON.

## Iteration64 complete short-live WASP kernel

68 boundary and16 representative-output checks PASS;243-244VGPR,no scratch/spill. Best representative361-427TOPS remains below fixed targets and earlier fast candidates. Correct controlled configuration only; no default promotion. See iteration_64_short_lifetime.md.

## Iteration65 batched A-fragment consumption

170 boundary and40 representative-output checks PASS;243-244VGPR,no scratch/spill. Best representative429-510TOPS remains below fixed targets. Correct controlled configuration only; no default promotion. See iteration_65_batched.md.

## Iteration66 bounded deferred A fragments

170 boundary and40 performance checks PASS. Batches1-8 no spill;16 spills and severely regresses. No representative gain over nondeferred65; no promotion. Correct controlled configuration only; no default promotion. See iteration_66_deferred.md.

## Iteration67 MMAC priority scope

306 boundary and72 performance checks PASS;no scratch/spill. Per-batch priority gives small shape-dependent gains; two controls remain faster. Correct controlled configuration only; no default promotion. See iteration_67_priority.md.
