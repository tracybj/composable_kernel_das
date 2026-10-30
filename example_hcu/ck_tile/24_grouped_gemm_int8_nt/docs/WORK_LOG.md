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

## Iteration68 overlapping A-read/MMAC

204 boundary and48 performance checks PASS;no scratch/spill. Representative best466/572/499/456TOPS, still below fixed targets. Correct controlled configuration only; no default promotion. See iteration_68_a_prefetch.md.

## Iteration69 one stage publication/reuse barrier

204 boundary and48 performance checks PASS;no scratch/spill. Representative best515/628/542/500TOPS; one of four exceeds its frozen target. Full344 and public1178 acceptance pending. Correct controlled configuration only; no default promotion. See iteration_69_one_barrier.md.

## Iteration70 original344 large-M screen launched

Freeze iteration69,6 one-barrier candidates plus current-default instance. Bounded whole-output validation. Running; no completion/target claim. See iteration_70_large_screen.md.

Iteration70 completed:344/344 original shapes,2408 numerical output checks PASS. Oracle {'shapes': 344, 'ge1': 109, 'ge05': 344, 'min_R': 0.7587955956334363, 'median_R': 0.9542826236407698}; same-build default {'shapes': 344, 'ge1': 11, 'ge05': 329, 'min_R': 0.45609053335500704, 'median_R': 0.6110435596582857}. No public promotion. See full CSVs/audit.

## Iteration71 dimension-only draft rules

Depth3/minleaf12 M/N/K/G draft tree; {'total': 344, 'ge1': 99, 'min_R': 0.7587955956334363, 'median_R': 0.9498076525128778}; regressions>3%=1. Not public acceptance or default promotion. Full-domain guards/fallback and public replay remain required.

## Iteration72 standalone WASP/fallback

114 boundary and24 performance checks PASS;no scratch/spill. Root headers have no experimental includes;214/232VGPR. Public registry integration follows. Correct controlled configuration only; no default promotion. See iteration_72_standalone_wasp.md.

## Iteration73 public WASP integration

3 public WASP instances;registry47. Host38+80,CPU1178 relabel/reverse and34 public full-range boundaries PASS. M/N/K/G-only large-M route with generic runtime stride fallback. All1178 public performance pending. See iteration_73_public_integration.md.

## Current public-v73 full1178 snapshot

{"numeric_pass": 1178, "ge1": 915, "ge05": 1178, "min_R": 0.7617497991099058, "median_R": 1.2049133250337323, "library_sha256": "cc5388e0e0d731a9775b8e4b2c4e8576ca62a4a284bc0f2cbdcd7449d086d16b", "phase": {"decode": {"total": 872, "ge1": 844, "median_R": 1.3274243393254848}, "prefill": {"total": 306, "ge1": 71, "median_R": 0.9398759626992711}}}
See current_public_v73_1178.csv; all rows share one library identity,all1178 newly measured.

Public73 gap analysis:915/1178 fixed targets met;263 remain (14 M<256,249 large-M). Original-target CSV preserved, no target relaxation. See current_public_v73_gap_summary.md and remaining.csv.

## Iteration74 cache-swizzle stride ablation

132 boundary and24 performance checks PASS;no scratch/spill. Generalized cache eligibility does not improve representative throughput; disabling regresses power-of-two strides. Original public policy retained. Correct controlled configuration only; no default promotion. See iteration_74_cache_stride.md.

## Iteration75 fresh long-M comparison and PMC

All6 fresh and2 profiled output/input/guard checks PASS. Fresh BLAS635.7TOPS vs public CK499.7TOPS. Complete G8 counter normalization uses8 BLAS versus1 CK launch. Higher CK VALU/LDS and TCC misses,lower accumulated LDS waits; test CTA order next. Profiled times excluded from rankings. See iteration_75_long_m_comparison.md.

## Iteration76 M-group CTA order

228 full-range and72 performance checks PASS; no scratch/spill. M-group1/2/4/5/8/16 alone does not close the large-M gap. Public selector and all1178 target/measurement CSV retained. See iteration_76_cta_order.md.

## Iteration77 structured-buffer record address decomposition

176 full-range and32 performance checks PASS;214VGPR,no scratch/spill. Independent quotient/remainder implementation improves representative K6144/K7168; regresses some power-of-two pitches. Validate paired same-allocation gains and all344 large-M domain before dimension-based promotion. Public CSV unchanged. See iteration_77_buffer_record_stride.md.

## Iteration78 same-allocation record1024 confirmation

All12 full-output/input/guard checks PASS. Fresh frozen BLAS635.6TOPS,publicCK500.2,record0control499.6,record1024 574.9. Representative improvement survives paired alternate order. Target remains636.6TOPS; no public promotion. See iteration_78_paired_buffer_record.md.

## Iteration79 all344 large-M record stride screen launched

176 full-range prerequisites and12 paired comparisons PASS. Five candidates per original shape,all dimensions and groups unchanged. Running; no completion claim. See iteration_79_large_record_screen.md.

Iteration79 completed:344/344 shapes,1720 output/guard/input-sample checks PASS. Transparent non-power-K draft {'total': 344, 'ge1': 156, 'min_R': 0.2967098624867015, 'median_R': 0.9938498117828563}; >3% regression count=0. Independent5-candidate timing outlier recheck PASS; original records preserved. Current public CSV remains v73. See iteration_79_large_record_screen.md and audit/CSVs.

## Iteration80 public record-stride integration

RecordCap0/1024 typed policy,stable IDs19004/19005,registry49. Host45+80,CPU1178 dimension replay/relabel/reverse and44 public full-range boundaries PASS. Five WASP kernels,no scratch/spill.168 input shapes select new policy. Formal all1178 performance pending;v73 formal CSV retained. See iteration_80_public_record_policy.md.

## Current public-v80 full1178 snapshot

{"numeric_pass": 1178, "ge1": 975, "ge05": 1178, "min_R": 0.7954511469116752, "median_R": 1.211549239005743, "library_sha256": "10f21a6d71ed11497c9a3c72b621d8e7a717cb3c7c7294fdc009f2576e1ee04e", "phase": {"decode": {"total": 872, "ge1": 846, "median_R": 1.329779528568767}, "prefill": {"total": 306, "ge1": 129, "median_R": 0.9915499524803606}}, "comparison_to_v73": {"previous_ge1": 915, "newly_ge1": 62, "lost_ge1": 2, "new_policy_shapes": 168, "new_policy_ge1": 92}, "by_M": {"1": {"total": 112, "ge1": 112, "median_R": 1.36935341541342, "min_R": 1.0361658763640191}, "2": {"total": 112, "ge1": 112, "median_R": 1.3677251304888212, "min_R": 1.034346934975136}, "4": {"total": 112, "ge1": 112, "median_R": 1.3727935033615677, "min_R": 1.0367730290958077}, "8": {"total": 112, "ge1": 112, "median_R": 1.3682783413311945, "min_R": 1.0359230721795651}, "16": {"total": 112, "ge1": 111, "median_R": 1.3246991879532222, "min_R": 0.9095332555794282}, "32": {"total": 104, "ge1": 102, "median_R": 1.3635964388440929, "min_R": 0.9751164835920785}, "64": {"total": 92, "ge1": 90, "median_R": 1.418472206556218, "min_R": 0.9441828365791097}, "128": {"total": 78, "ge1": 69, "median_R": 1.2013337617988888, "min_R": 0.8641217615658141}, "256": {"total": 38, "ge1": 26, "median_R": 1.0913950300108366, "min_R": 0.9580386643474805}, "2048": {"total": 108, "ge1": 66, "median_R": 1.0231299568943282, "min_R": 0.7954511469116752}, "4096": {"total": 88, "ge1": 49, "median_R": 1.0242692539004492, "min_R": 0.8890193284509845}, "8192": {"total": 72, "ge1": 14, "median_R": 0.9709951193604038, "min_R": 0.9217822527123688}, "16384": {"total": 38, "ge1": 0, "median_R": 0.9440490509612411, "min_R": 0.8953672531051114}}, "remaining": {"total": 203, "small_M_lt256": 14, "large_M_ge256": 189}}
See current_public_v80_1178.csv; all rows share one library identity,all1178 newly measured.

## Iteration81 A-batch/scalar-output controls

352 full-range and112 performance checks PASS;no scratch/spill. Batch4/vector remains best on major large-M representatives;larger batches/direct scalar output do not close the gap. Keep public v80 unchanged. Geometry/orientation observations and512x256x64 next-structure hypothesis in iteration_81_output_batch.md.

## Iteration82 paired small-M recheck

All12 whole-output/input-clone/guard checks PASS. v73/v80 match on identical allocations for both apparent small-M regressions;original formal v80 records and975/1178 count retained. Timing noise cause unproven. See iteration_82_paired_small_m.md.

## Iteration83 wide-M structural trial

216 full-range and56 performance checks PASS.512x256x64,twelve waves,four producers:all8 kernels spill and representative throughput collapses. No public promotion. Next compare an eight-wave cooperative allocation with the same mathematical tile. See iteration_83_wide_m.md.

## Iteration84 cooperative wide-M structure

216 full-range and56 performance checks PASS. Eight cooperative waves do not remove wide-accumulator spill/performance cost. No public promotion. Next trade N128 for128 accumulator values and changed A/B reuse;see iteration_84_cooperative_wide.md.

## Iteration85 narrow-N wide-M structure

216 full-range and56 performance checks PASS;156--173VGPR,no scratch/spill.512x128x64 halves B requests but increases A movement/LDS/stage costs;representatives remain slower than public v80. No promotion. Return to fast256x256x128 consumer-schedule investigation;see iteration_85_narrow_n.md.

## Iteration86 single-publication C/CP schedule

360 full-range and84 performance checks PASS. CP variants spill;all-C controls remain fastest/no spill. Production assertion and formal v80 CSV retained. Next pair early/late B prefetch within all-C. See iteration_86_consumer_schedule.md.

## Iteration87 early B prefetch and generated wait analysis

240 full-range and56 performance checks PASS;214VGPR,no spill. All28 early/late pairs regress. Own ISA shows future loads issued before an additional generated first-MMAC vmcnt(0),blocking intended overlap. Next factorial builtin/asm wait experiment;production v80 retained. See iteration_87_early_b.md.

## Iteration88 compiler-visible wait

480 full-range and112 performance checks PASS;214VGPR,no spill. Builtin removes redundant generated VMEM wait but no stable material performance gain. Keep v80. Shift structural analysis to provided DeepGEMM FP8. See iteration_88_compiler_wait.md.

## Iteration89 four-wave N reuse

248 full-range and84 same-build performance checks PASS;180/204VGPR,16KB LDS,no spill. No gain over WASP controls on seven measured shapes. Next isolate physical weight packing with conversion costs. See iteration_89_n_reuse.md.

## Iteration90 physical B packing

96 full-range,42 screen and28 rotating-order output/guard checks PASS. Packed198VGPR,no spill. Original-weight packed kernel improves6/7 against same-build best raw controls;GPU conversion cost and reuse threshold explicit. No raw-NT production promotion. See iteration_90_packed_b.md and deepgemm_fp8_to_int8_review.md.

## Iteration91 raw NT A+B LDS

124 full-range and56 performance checks PASS;151/175VGPR,64KB LDS,no spill. Native overlap visible in ISA,no extra pre-MMAC VMEM wait. All seven slower than same-build WASP controls;keep v80. See iteration_91_lds_operands.md.

## DeepGEMM raw FP8 cross-check

Verified raw-NT fp8_gemm_hipc is256-square4P8C/Cprime WASP,not universally packed;fp8_gemm is a hipBLASLt wrapper;ASM grouped CU is a code-object launch wrapper. Distinguish these from the packed grouped variants in the review. No new GPU claim or production change.

## Iteration92 raw NT linear B addresses

300 full-range and70 performance checks PASS;214VGPR,no spill. Seven representative best-linear candidates beat same-run best-structured by1.4–4.3%,no physical conversion. Broaden to344 before promotion. See iteration_92_linear_b.md.

## Iteration93 M128 domain recheck

All74 native-eligible originalM128 shapes,370 output/guard checks PASS. Four-wave Nreuse doesnot close any ofnine prior M128 misses;keep6042. Four already-passing1232-tail cases excluded explicitly. See iteration_93_m128_recheck.md.

## Iteration95 native-domain resources and scope confirmation

160 legal-domain full-range and84 performance checks PASS;198–201VGPR versus214 full kernel. Performance doesnot justify incomplete-domain promotion. User permits one-time static INT8 weight packing/reuse;FP16/BF16 existing ABI/layout/defaults must stay unchanged. Plan additive INT8-only cached API and344-shape comparison.

## Iteration96 all344 large-M packed/raw screen launched

User permits one-time INT8 weight preparation/reuse;FP16/BF16 ABI/layout/defaults protected. Freeze v90/v92,6 candidates per original344 shapes,2064 expected full-output records. Running,not final acceptance.

## Iteration96 validator memory repair

Verified process terminal afterOOM inboolean comparison reduction atID464.132 complete shapes/792 PASS records preserved. Bounded256-row validation,unchanged GEMM workload/libraries/timing;resume audited atID464. Original error/log preserved.

## Iteration96 completed all344

2064 full-output records PASS;prior792-prefix hash preserved after validator-only OOM repair. Same-run control168/344,oracle284/344,dimension-only draft284/344,minR0.9054,no>3% regressions.60 large-M misses remain. No formal1178 update. Additive INT8-only API integration next;existing16bit ABI/layout/defaults protected.

## Iteration99

New isolated INT8 cached-weight API all1178 numerical PASS;1093/1178 reach frozen BLAS targets,minR=0.866067,medianR=1.215079;85 remain.46full-range and8weight/stream-lifetime checks PASS. No existing FP16/BF16 source/ABI/default changes. Preparation time/storage separate. See iteration_99_cached_api_1178.md and current_cached_v97_1178.csv.

## Iteration100

Independent M128 WASP(2producer waves,N64/128/256,LDS32KiB)built;212full-range checks and194performance outputs PASS. Same-build screen closes7/14 selected gaps;M128 packedN128 improves five samples1.70--2.00x,while large long-K still prefers currentM256 packed. No production/API/16bit change. See iteration_100_split_m.md/JSON/CSV.

## Iterations101/102

All422 M>=128 requirements screened(1523 numerical PASS),plus all38 M256(152 PASS). Dimension-only draft reaches365/422 in calibration,all78M128 reach target(including4 unchangedtails),zero>3%regressions. No production promotion yet;formal1093/1178 unchanged. See iteration_101_102_policy.md and CSV/audits.

## Iteration103 integration gate

Independent INT8 routes6/7 integrated by M/N/K/G only;K512/768 packing enabled;47full-range and16shared-weight/stream checks PASS;CPU ABI/oldsymbol-address/sourcehash checks PASS. Actual library owner verified bydladdr. New full1178 acceptance pending;old1093/1178 formal record unchanged. No FP16/BF16 source/default edits.

## Iteration103

New isolated INT8 cached-weight API all1178 numerical PASS;1117/1178 reach frozen BLAS targets,minR=0.944792,medianR=1.252105;61 remain.47full-range and16weight/stream-lifetime checks PASS. No existing FP16/BF16 source/ABI/default changes. Preparation time/storage separate. See iteration_103_cached_api_1178.md and current_cached_v103_1178.csv.


# Iteration104:wave-local packed B packets

Tested static INT8 weight format2 `[G,K/128,N/32,2,2,4,16,16]`. A wave consumes one contiguous4KiB packet, using one vector byte address and scalar K-stage offset. Same M256/N256/K128 geometry,4 producer and8 consumer waves,64KiB LDS,and the existing all-consumer publication protocol. Both formats read identical bytes and unique pages per CTA;this is not a claim of halving TLB misses. No runtime case-ID dispatch.

CPU byte-bijection/per-wave address model PASS.48 full-rangeINT8/FP64 checks PASS,including padded strides,ragged/empty groups,tails,input immutability,and output guards.16 representative original shapes,64 performance records PASS;balanced order,2 warmups,5 repeats,3-sample median. Pack preparation is recorded separately under the approved static-weight-reuse contract.

New batch4 uses197VGPR versus198 for format1,with no spills;batch2 uses181 andbatch8 uses229. All use64KiB LDS. LongK batch4 gains are mostly below0.4%;batch2 andbatch8 regress on important longK cases. M16384/N4096/K7168/G8 remains647.56TOPS versus647.66 control,below675.51 target. M16384/N4096/K6144/G8 is628.59 versus628.23 control,below636.59 target. Packet continuity alone is insufficient,so format2 is not promoted and no full344/1178 rerun is justified for this candidate.

The control is the same-build old M256 packed route,not the entire v103 selected policy. ShortK cases already served by M128 in production must not be described as gains over production. Current production remains v103:1178 numericalPASS,1117 meet frozen BLAS targets,61 remain. Next compare a distinct M64 geometry and further longK dataflow/scheduling alternatives.


# Iteration105:smallM64 producer/consumer geometry

Hypothesis:the legacy M64 route does not exploit static INT8 packed weights,and M128 would waste row work. Added separate CK-Tile Problem/Pipeline/Kernel/Epilogue,64x(64/128/256)x128 tiles,one A producer wave and2/4/8 consumers. Each wave owns64x32 output with32INT32 accumulators. A uses two8KiB slots(total16KiB LDS),same established publication/reuse barriers. B retains format1 and direct MMAC fragments;activations remain raw. No FP16/BF16 interface changes.

CPU A-address model PASS.108 full-rangeINT8/FP64 checks PASS,including native and odd K-stage counts,padding,ragged/empty groups,tails,output guards,and full input immutability. All9 kernels use86VGPR,no spill,16KiB LDS. All92 original M64 requirements,884 same-build measurements PASS,2warmups/5repeats/3sample median and rotated order. Four requirements withN1232 orK1232 cannot use native candidates and retain legacy output checks.

The broad N64/A2 packed candidate has minimum speedup0.9964 and median1.1613 on88 native shapes. N256 can regress on other shapes,so it is restricted by dimensions toN>=4096,K>=4096,G<=12;remaining native M64 usesN64/A2. This dimension-only draft reaches all92 targets in the development screen. It is calibrated on this screen and must pass independent public-API acceptance before production claims.

Previously failing M64/N4096/K6144/G8:108.50 to148.39TOPS,target114.48. M64/N4096/K7168/G8:110.38 to149.84TOPS,target116.60. These are new-run comparisons,not replacements of v103's frozen records. Main public API remains v103 until integration gates and full1178 measurements finish. Next integrate additive routes8/9 using unchanged packed format and ownership/lifetime contract.


# Iteration106:integrate M64 packed routes into the isolated INT8 API

Add routes8(M64xN256,A-batch2)and9(M64xN64,A-batch2)under `cached_weights/m64/`. UniformM64 native plans become eligible. Route8 requiresN>=4096,K>=4096,G<=12;all other native M64 usesroute9. The selection uses only dimensions and availability. Ragged/empty and unsupported N/K preserve the existing provider. Both routes explicitly bind owned packedB using existing format1;no new weight layout or activation packing is introduced.

Build in `build/cached_v106/` passes. Exactly8 new C API symbols are exported,zero old grouped-GEMM ABI owners are added. C structs/tags/offsets and all9 old public-context source hashes are unchanged;original INT8 host45 tests pass. Same-process loading preserves all4 old C entrypoint addresses. This is ABI/loader evidence,not new FP16/BF16 numerical or performance evidence.

55 full-rangeINT8/FP64 boundary checks PASS with route coverage0/1/3/4/5/6/7/8/9,including M64 native packed routes,wide padding,odd stage count,ragged/empty fallback,input immutability,packed-byte checks,guarded output and nondefault preparation/compute streams.20 shared-weight tests PASS across M64/128/256/512/2048,including explicit source-address refresh,two generations,two streams,and queued-work destruction. CPU input/CSV/resume corruption checks for1178 shapes PASS.

The dimension policy was calibrated on iteration105's92-case screen and is not independent acceptance. A fresh full1178 run is required. Existing formal v103 record remains1178 numericPASS,1117 reach BLAS,61 remain until the new public-API run finishes and is audited. Pack preparation and persistent bytes are reported separately under the approved reusable static-weight contract. Full target completion remains unproven.

## Iteration106

New isolated INT8 cached-weight API all1178 numerical PASS;1115/1178 reach frozen BLAS targets,minR=0.957414,medianR=1.269414;63 remain.55full-range and20weight/stream-lifetime checks PASS. No existing FP16/BF16 source/ABI/default changes. Preparation time/storage separate. See iteration_106_cached_api_1178.md and current_cached_v106_1178.csv.


# Iteration107:more consumer waves sharing M256 A publication

Keep packed format1,M256/K128,A's two32KiB slots,and the all-consumer publication protocol. Increase N256 toN384,with12 consumer and4 producer waves(1024threads). Each consumer still owns256x32 output/128INT32 accumulators. For divisible N,total B loads and MMAC work are unchanged,but A publications/CTA count decrease by one third. Native N multiples32 support a partial final block by skipping out-of-N consumer epilogue stripes;buffer resource bounds protect unused B reads. M/stride diagnostic fallback remains complete. MGroup8/16 variants separately probe cache scheduling.

The initial C12/A4 variants39001/39008 spill:192VGPR,440private bytes,241VGPR spill metadata count. They pass single-stageK128 but fail multi-stageK384 numerically. A three-shape diagnostic finds these two fail while C12/A2 and C8 controls pass. This is correlation with spilling,not a proved root mechanism. Both are quarantined entirely before performance measurement;their code is experimental and must not be selected. First failure and diagnostic logs are preserved.

Remaining control plus4 candidates pass60 full-rangeINT8/FP64 checks,including odd stages,padding,ragged/empty groups,N tails,guards and input immutability. C12/A2 uses182VGPR,zero spill,64KiB LDS. All80 representative performance outputs PASS on16 original shapes,with balanced order,2warmups/5repeats/3sample median. It is a development screen,not public-API acceptance.

Valid N384/A2 improves important longK cases: M16384/N3072/K4096/G48 goes570.36 to611.46TOPS(R1.06346);M16384/N4096/K6144/G8 goes625.29 to651.01(R1.02266);M8192/N6144/K7168/G12 goes644.25 to685.07(R1.04602). M16384/N4096/K7168/G8 improves647.68 to672.83 but remains below675.51 target(R0.99604). Not universal:shortK regresses,andM4096/N1536/K2048/G8 drops2.36%. Cache MGroup8/16 variants do not improve the tested longK cases. The control is the same-build old M256 packed route;shortK production already uses M128,so these are not universal gains over current policy.

All344 M>=256 original shapes are now screened with just the old M256 and valid N384/A2 candidates. No production promotion or target-completion claim until the broad screen defines a dimension policy and a fresh public-API validation passes. The current formal report is v106:1178 numericPASS,1115 targets,minimumR0.957414. M64's92 cases all meet target. Additional exact device-code inspection finds all36 shared kernels byte-identical between v103/v106;only2 M64 kernels are added. This is not a paired timing or equal-cache-placement claim.


# Iteration107 broad result and iteration108 M32 controls

All344 original M>=256 requirements completed through the v107 same-build screen:688 outputs PASS. OldM256 reaches220 targets;wideN384/A2 reaches267;per-shape oracle281. These are not current-policy attainment counts:94 requirements currently use other raw/transposed/M128 routes. On the250 requirements whose current production route is oldM256 packed,controls reach196 targets,wide reaches241,and the oracle245. This oracle cannot be deployed as a case-ID table. A dimension-only selection and fresh public-API acceptance remain required. Raw measurements are in iteration_107_all344_raw.csv.

Wider N can regress when task supply is small. Examples M2048/N3072/K4096/G4 andM4096/N3072/K4096/G4 remain below target;N2048/K4096 also often prefers the old route. K2304 cases may regress. LongK large workloads improve consistently,butM16384/N4096/K7168/G8 andG12 remain marginally below target. Preserve existing non-route5 paths until separately compared. No public default was changed by this screen.

Iteration108 checks all104 original M32 requirements against all3 existing instances7501/7502/7504.312 outputs PASS;current and per-shape oracle both reach102 targets. M32/N3072/K4096/G4 tops at51.2803TOPS(R0.975166),andM32/N4096/K7168/G8 at63.5415(R0.995705). Existing-instance retuning cannot meet all targets. A packed M32 geometry or another dataflow optimization is needed;no threshold relaxation or per-ID fallback is used.

Next investigate empty-consumer work in partial N384 blocks. The valid kernel currently skips out-of-N stores but still issues B reads,LDS A reads and MMAC in those waves. For N4096,each final N384 block has only8 valid consumers out of12. Proposed change:unused consumer waves skip loads/MMAC while still participating in every publication/reuse barrier. This is an untested design,not a correctness or performance claim. It may recover the remaining sub-percent longK gap;register/spill and barrier correctness must be verified. A true M32 packed pipeline can reduce producer slab32rows,accumulators16/lane andLDS8KiB,with independent full-range and104-case validation.

Formal public API remains v106:1178 numericPASS,1115 targets,all R>=0.95,M64 all92 targets. Changes to source policy require new formal measurements. Existing FP16/BF16 ABI/default sources remain unchanged.


# Iteration109:skip empty N384 consumer work while retaining barriers

Hypothesis:the last N384 block can contain unused32-column consumer stripes. Skip their initial/next B reads,LDS A reads andMMAC while retaining every publication/reuse barrier;the existing epilogue already skips their stores. No producer or consumer returns before the final shared synchronization. Other geometry,packed format1 and MGroup4 stay fixed. Added independent Problem/Pipeline/Kernel/Epilogue in wasp_masked_n_v109. The control's four v107 headers match the prior compiled-source hashes.

Build succeeds:both old and masked use182VGPR,zero spill,64KiB LDS;masked uses32 rather than30SGPR.24 full-rangeINT8/FP64 checks PASS,including odd stages,N tails with mostly-empty consumers,padding,ragged/empty groups,guards and immutable inputs.16 representative shapes/32 performance outputs PASS,balanced order,2warmups/5repeats/3sample median.

Some shortK cases improve: M16384/N4096/K1536/G4 goes401.00 to409.48TOPS(+2.11%). However longK gains are small or absent: M16384/N4096/K6144/G8 goes651.37 to650.92;M16384/N4096/K7168/G8 goes672.37 to671.08,remaining below675.51 target. Fully-divisibleN cases also move slightly,so the timing differences cannot be attributed exclusively to skipped tail work without further ablation. The hypothesis does not close the remaining longK gap. Do not promote or launch a full344/1178 run for this candidate.

Continue with true M32 packed geometry to address the two decode gaps,and use the validated unmasked N384/A2 as a separate basis for dimension-selected largeM improvements. A fresh public-API acceptance remains required. Current formal v106 remains1178 numericalPASS,1115 targets;no FP16/BF16 interface or defaults changed.


# Iteration110:true M32 packed WASP geometry

Added independent Problem/Pipeline/Kernel/Epilogue. One producer publishes32rows of raw A per128K stage using4 vector loads per lane into4KiB;double buffering uses8KiB LDS. Each consumer owns32x32 output with16INT32 accumulators. N64/128/256 uses2/4/8 consumers. Packed B retains approved format1;raw B variants isolate layout from geometry. Input activations are not packed. No FP16/BF16 interface/default changes.

CPU producer/consumer byte-address model PASS.96 full-rangeINT8/FP64 checks PASS,covering odd K-stage counts,padding/wide strides,M/N tails,ragged/empty groups,input immutability and guarded outputs. All8 kernels use62VGPR,zero spill,8KiB LDS. All104 original M32 requirements,904 performance outputs PASS,2warmups/5repeats/3sample median,balanced order. Four non-native N/K requirements retain legacy in the screen. Several highG cases prefer legacy,so packed M32 is not universally selected.

Dimension-only draft:when packed weights are available,K>=3072 andG<=12,useN256/A2 ifN>=4096 andK>=4096,otherwiseN64/A1;keep legacy elsewhere. It promotes11 requirements,reaches all104 targets,and has no same-run regression in this screen. This is calibrated development evidence,not independent acceptance.

M32/N3072/K4096/G4:52.3198 to76.7836TOPS(+46.76%),target52.5863. M32/N4096/K7168/G8:63.7052 to77.9300TOPS(+22.33%),target63.8156. Raw versions largely fail to reproduce packed gains,supporting the value of contiguous prearranged B fragments in this geometry. The exact attribution still includes coupled changes to geometry and staging versus the legacy kernel.

Next add explicit INT8-only routes with the same packed-weight lifetime contract,then validate full public API. Also compare the wideN candidate against each actual current route before formulating a largeM selection. Formal v106 remains1178 numericPASS,1115 targets until fresh acceptance completes;do not substitute104-case draft counts for full1178 results.


# Iteration111:actual current-route comparison and CTA-wave selection

Earlier M256-only controls do not represent the94 shapes already served by M128 or transposed/raw routes. This same-module adapter reuses the frozen v106 private dispatcher for every actual selected route and compares it with valid N384/A2. Probe IDs encode implementation routes,not requirement case IDs. Production selection is not modified by the adapter. Control dependencies and old context are frozen in build/v111.

150 full-rangeINT8/FP64 checks PASS across9 old routes and the wide candidate,including native/fallback/tails/empty groups/padding/guards. All344 original M>=256 requirements complete688 numericalPASS records. Controls reach286 targets,wide alone268,per-shape oracle336. A dimension-only wave policy reaches the same336,with220 promotions andminimum same-run speedup0.99609. It is calibrated on this data;only a fresh public-interface full1178 run can establish delivery performance.

PhysicalGPU7 runtime reports80 active CUs,196608 registers perCU,2560 max threads perCU,64-lane waves,and8MiB L2. KFD independently reports320SIMDs,4SIMDs perCU,10max waves perSIMD,and64KiB LDS. Both oldM256 and wideM256 use64KiB perCTA,so CTA wave quantization matters. Letpadded_M_tiles=round_up(ceil(M/256),4),base_CTAs=padded_M_tiles*ceil(N/256)*G,wide_CTAs=padded_M_tiles*ceil(N/384)*G. Divide each by device CU count and round up. For original route1/5 andK>=1024,select wide if base_waves/wide_waves>=1.47,or if base_waves>=24 andratio>=1.44. Thresholds are measured dispatch calibration,not universal hardware laws. Other routes stay unchanged. Runtime uses only dimensions/device properties;no target spreadsheet,case-ID table or timing lookup.

ForM2048/N1536/K2048/G4,the old grid192CTAs takes3 waves versus wide128CTAs/2waves;wide closes its gap. ForM2048/N3072/K4096/G4,the old384CTAs takes5waves versus wide256CTAs/4waves;the widened CTA's greater work makes it unfavorable. This distinguishes shapes without enumerating their IDs. N filling and kernel setup costs remain imperfectly modeled,so the result does not claim all future shapes are optimal.

Eight measured gaps remain after this largeM draft:threeK768 long-M cases,M8192/N1536/K2048/G4,twoN3072/K4096 small-task cases,andtwoM16384/N4096/K7168 cases. Independently validated M32 packed routes can address the two decode gaps. Next integrate those with this wave policy,verify lifecycle/ABI,and run new full1178 acceptance. Formal current report remains v106:1115 targets,1178 numericalPASS. No FP16/BF16 source or ABI/default changes.


# Iteration112:integrate M32 packed and device-aware M256/N384 selection

Add INT8-only routes10(M32/N256/A2),11(M32/N64/A1),12(M256/N384/A2). M32 is eligible only for packed weights,K>=3072,G<=12;N>=4096 andK>=4096 uses10,otherwise11. Other M32 cases retain legacy. NewM32 headers are undercached_weights/m32 with a validA2 default. Wide headers use a distinct private namespace underwide_n;known spillingC12/A4 and the unhelpful109 mask are not selected or instantiated.

The old selector is retained as the base. Only base routes1/5 withM>=2048,K>=1024 may use12. Padceil(M/256)toMGroup4;estimate old/wideCTA counts usingceil(N/256)andceil(N/384),multiplyG,then round up bydevice CU count. Select12 forwave ratio>=1.47,or baseline>=24waves andratio>=1.44. CU count is saved from the device properties already queried atweight creation,adding no new device-query call. Thresholds are measured calibration,not universal limits. Runtime does not access requirement IDs,phase,target tables or timing records.

All three routes explicitly bind the same owned packed format1. Weight ownership,ready event,stream dependencies,source immutability and plan destruction semantics stay the same. One prepared expert-weight set can serve differentM plans. All opaque handles remain library-instance specific. No FP16/BF16 descriptors,entrypoints,defaults or existing source files are modified.

Build passes with exactly8 INT8 C API entries andzero old ABI owners. All9 old public-context hashes are unchanged;mixedC ABI/tag/offset checks andoriginalINT8 host45 tests pass. Same-process loading preserves all4 old C entrypoint addresses. This is ABI/loader evidence,not a new FP16/BF16 GPU benchmark. The3 integrated kernels match expected resources:M32 uses62VGPR/8KiB LDS,wide uses182VGPR/64KiB LDS;allzero spill/private bytes.

65 full-rangeINT8/FP64 checks PASS,covering every selected route0/1/3/4/5/6/7/8/9/10/11/12,padding/wide raw strides,odd stages,ragged/empty groups,packed bytes,immutable inputs,guarded outputs andnondefault streams.32 lifetime checks PASS acrossshared weight plans,two generations ofexplicit same-address refresh,two streams,and destruction afterqueued work,including newroutes10/11/12.11 CPU selector checks verifydimension-only choice,device-CU dependence andunknown-device fallback. CPU1178 contract/CSV/resume-corruption checks PASS.

Fresh full1178 public-interface acceptance is still required. DevelopmentM32 policy reaches104/104,andlargeM policy336/344,butthese are calibrated screens,not delivery counts. Previous formalv106 remains1178 numericalPASS/1115 targets until the newrun is complete and audited. Pack preparation time/storage remain separate underthe approved static-weight reuse contract. All-target completion remains unproven.

## Iteration112

New isolated INT8 cached-weight API all1178 numerical PASS;1170/1178 reach frozen BLAS targets,minR=0.982297,medianR=1.276317;8 remain.65full-range and32weight/stream-lifetime checks PASS. No existing FP16/BF16 source/ABI/default changes. Preparation time/storage separate. See iteration_112_cached_api_1178.md and current_cached_v112_1178.csv.


# Iteration113:balanced-resource tiles and shortened A live ranges

Tested M128/N192 with6consumers/2producers atA4 andA2;M256/N320 with10consumers/4producers atprefetchedA2 ornon-prefetchedA4;andM256/N384 with12consumers/4producers atnon-prefetchedA4. N-stripe output guards preserve partialN correctness. All reuse format1,existing publication/reuse barriers,andraw A. Current actualv112 routes are compiled in the same module;all source dependencies are checked before/after build.

255 full-rangeINT8/FP64 checks PASS across12 current routes and5 candidates,including odd stages,padding,N tails,ragged/empty groups,guards andimmutability. NewM128 A4/A2 use134/118VGPR and32KiB LDS. The3 M256 candidates use182VGPR and64KiB LDS. Allzero spill/private bytes.20 representative original shapes,105 timed outputs PASS,including all8 remaining targets.2warmups/5repeats/3sample median androtated order.

None ofthe8 remaining targets improves overthe current path. ForM16384/N2048/K768/G4,the current253.01TOPS remainsbest;newM128/N192 is about6.4%slower. ForM16384/N4096/K7168/G8,current672.22 remainsbest;non-prefetchedA4/N384 is about8.1%slower. M256/N320 variants also regress. An already-passing M128/N3072/K4096/G4 improves213.72 to232.88TOPS,butthis doesnot solve a missing target. No full422/1178 rerun or default promotion is justified.

Lower registers andapparently balanced block sizes didnot predict improvement. Per-SIMD wave placement,copy/compute request schedules and cache traffic need stronger evidence;this experiment doesnot prove anoccupancy explanation or hardware ceiling. Preserve the validated v112 policy. Next explore a differentrole assignment:cooperative A publication bycompute waves with a widerN output tile,usingpacked B andM256 reuse. Prior raw-B cooperativeM128/K64 failure doesnot rule outthis differentgeometry/dataflow combination. Its synchronization andresource costs must be tested from scratch.

Formal report remainsv112:1178 numericalPASS,1170 targets,all872decode pass,and8prefill gaps. ExistingFP16/BF16 source/default/ABI remainsunchanged. See current_cached_v112_1178.csv anditeration_112_remaining.csv.


# Iteration114:cooperative packed M256 A-copy and MMAC waves

Replaced dedicated producer waves with8 or16 compute waves that also copy disjoint A slabs into the same XOR LDS layout. Two32KiB slots preserve M256/K128 reuse. OutputN256 andN512,format1 packed weights,raw dynamicA,INT32 output,no scale. Each wave resets m0 after VMEM-to-LDS issue before DS reads. Publication and reuse wait allVMEM/LDS work at a block barrier. CPU byte-level A mapping and consumer fragment audit PASS.

Native N256/N512 kernels compile with185/183VGPR respectively,64KiB LDS,zero private bytes or spills. Static ISA includes realVMEM-to-LDS,MMAC and3m0 restore sites;counts are not dynamic hardware counters.210 full-rangeINT8/FP64 checks PASS across12 current routes and2new candidates,withsingle/odd/even/longK,padding,ragged/empty diagnostic fallbacks,guards andimmutable inputs.20original shapes/50 outputs PASS in balanced same-module2warm/5repeat/3sample median development timing.

N512 improves M2048/N3072/K4096/G8 from544.36 to550.33TOPS(+1.10%,R0.99909),andM4096/N3072/K4096/G4 from546.04 to550.31TOPS(+0.78%,R0.99892). Neither clears its frozen target. M16384/N2048/K768/G4 regresses about15%;M16384/N4096/K7168/G8 about2.2%. N256 cooperative is slower onall tested native shapes. Do notpromote or rerun full1178 withthis version. These results do not ruleout cooperative designs withdifferent M reuse,smaller LDS or B-prefetch timing.

Two development issues were corrected andtheiroriginal evidence retained. DTK rejected inline-assembly operands captured only inside astatic_for lambda;an unrolled ordinary loop compiles. The first performance runner exitedbefore anyGEMM:an overbroad test-count text replacement changed the sampling denominator255 to210,creating out-of-boundsPyTorch index_elementwise accesses. Corrected by targeted count replacement andCPU-generated indices withbounds assertions. Original failure log remains results/cooperative_packed_v114/run.log;valid measurements are in results/cooperative_packed_v114_fixed. The kernel binary andits numerical proof areunchanged bythe runner fix. C++source hashes arechecked againstthe build snapshot;correctedrunner hash is recorded separately.

Production remainsv112:1178numericalPASS,1170targets,8prefill gaps;all872decode targetsPASS. No FP16/BF16 changes. Next test M128/cooperative A publication with32KiB LDS andan early-B variant ofM256/N512;both needfresh resource,numerical andpaired evidence. Current v112 is preserved.


# Iteration115:M128 cooperative A publication and earlier B prefetch

Native M128/N128,N256,N512 cooperative copy-compute variants use32KiB LDS and64INT32 accumulators/lane. Compare late versus early next-B onM128/N256,andearlyB onM256/N512 againstv114 late-B evidence. Format1 andrawA unchanged;full-stage publication/reuse barriers remain. CPU byte mapping PASS. M128 uses118-121VGPR/32KiB LDS;M256183VGPR/64KiB;allzero spill/private.255 full-rangeINT8/FP64 checks PASS.20representative original shapes/115timed outputs PASS,includingall8 remaining frozen targets. Same-module controls,rotated2warm/5repeat/3sample median development screen.

Every tested shape retains the current selected v112 path asbest. M16384/N2048/K768/G4 current253.69TOPS;M128/N128 earlyB isabout6.8%slower,M128/N256 about16%slower. M2048/N3072/K4096/G8 current544.17TOPS;M128/N256 lateB about0.7%slower,earlyB about3.4%slower. M16384/N4096/K7168/G8 current673.42TOPS;M128/N256 lateB about4.4%slower,earlyB about21%slower. Earlier B andlower resource declarations do not establish better latency hiding or residency. Do notpromote.

The native M128 small-M examples have fewer validCTAs than theirpadded launch grid,sothe small-M regressions alone do not isolate the instruction path. Large-M controls also regress,which rulesout this particular configuration asareplacement. These experiments refute their predictions,notall cooperative architectures or ahardware ceiling. Stop further same-family tile scanning;inspectactual wait/issue andmemory evidence ofthe current production routes before choosing another rewrite.

Production remainsv112:1178numericalPASS,1170targets,all872decode targetsPASS,and8prefill gaps. Original FP16/BF16 source/default/ABI unchanged. No target substitution or tolerance relaxation. Raw paired data,resource metadata andfull numerical checks accompanythis log.


# 第116轮：剩余8条的原始BLAS赢家反汇编与同输入回放

## 目标与身份

从原始 `BlasLt_ggemm_tuning_baseline.xlsx` 的 `int8_NT!K` 逐条核对最大吞吐，与20261027双后端tuning原始CSV一致。全部8条的M/N/K/G和最大值完全匹配。保留原始目标，不重新tuning。使用原始bridge与BLAS二进制SHA验证后的固定算法，在物理卡7（PCI0000:f5:00.0）回放。当前CK比较器是v114模块内编入的实际v112选择路径，未启用v114实验候选。

| 用例 | 原始M/N/K/G | Max赢家 | 算法 | BLAS内部tile | 原始目标TOPS | 本轮BLAS TOPS | 本轮CK TOPS |
|---|---|---|---|---|---:|---:|---:|
|100|16384/2048/768/4|matmul|2972|192×256|257.837|257.635|253.227|
|911|2048/3072/4096/8|groupgemm|6950|256×256|550.831|549.751|545.534|
|920|4096/3072/4096/4|groupgemm|6950|256×256|550.902|551.537|546.937|
|96|16384/2048/768/8|matmul|2972|192×256|258.160|257.278|257.046|
|1126|16384/4096/7168/8|matmul|2979|256×256|675.507|675.794|673.525|
|92|16384/2048/768/16|matmul|2972|192×256|258.329|257.766|258.024|
|1130|16384/4096/7168/12|matmul|2979|256×256|675.932|676.033|674.563|
|704|8192/1536/2048/4|groupgemm|6950|256×256|426.956|425.096|427.269|

所有8条×2后端×3样本，共48次完整输出、完整A/B输入及输出保护区检查PASS。INT8输入[-8,8]，INT32输出，无scale；FP32参考关闭TF32，该数据域内整数结果精确。无profiler计时：10warm/50repeat/3sample median，交替后端顺序，共用相同原始A/B/C分配，CK仅使用其既有可复用packed B。准备不计入稳态时间。本轮704超过固定目标属于局部复测，不能替换原正式1178统计；92同轮CK超过回放BLAS但仍低于冻结目标，仍算未达标。正式结果保持v112。

## 反汇编方法与纠正

三个赢家都在本次BLAS安装的 `TensileLibrary_gfx936.co`，保存object和ISA SHA。只用llvm-objdump的 `--disassemble-symbols` 会在下一局部label截断；其ELF全局FUNC的size又为0。按同text段下一GLOBAL FUNC地址给出显式start/stop区间，取得完整计算体（末尾包含对齐padding）。最初短输出不是包装kernel/函数调用的证据，该解释已纠正。2972/2979/6950分别约51.7/66.6/78.8KB地址跨度，完整原始ISA仅留build及已授权本地分析副本，不提交供应商指令块。

## 已证明的结构

1. 三者都是768线程，即12个wave。wave_id<=7进入MMAC，后4个wave专做VMEM-to-LDS。8个计算wave按wave_id&7是否>=4分两批，交错低/高K半段。主干包含原生 `v_mmac_i32_16x16x32_i8`，不是由kernel名字推测的另一种INT8指令。
2. 生产者用buffer_load_dwordx4直接写LDS，两个32KiB地址槽交替。2972每个生产者发6个16B/lane向量，合计192×128字节；2979和6950每个生产者8个向量，合计256×128。6950名字中的MT...x32不能当作实际stage K：主干搬运、地址增量与双K64半段对应K128。
3. 输入指针已经按metadata及bridge映射核实。BLAS内部A指针位于matmul kernarg偏移32，对应原始B权重；内部B偏移40，对应原始A激活。生产者经s32构造的descriptor读内部A，消费者直接读内部B。因此BLAS在计算C转置，内部M/N对应原始N/M。输入和输出没有外部转置拷贝。2972内部192×256对应原始输出256×192；2979/6950为256×256。
4. MMAC半段开始前一次读齐12（2972）或16（2979/6950）个16B A片段，再等待LDS和连续发射MMAC。循环内可见下一B寄存器槽加载与A片段读取交叉，并用两批消费者及barrier安排依赖。MMAC段切换优先级1/0。三者声明255VGPR，实际launch资源粒度256，64KiB LDS，scratch0。该声明不是高occupancy证明。
5. 2979可见对原始物理pitch的record1024分解与除5余数计算的CTA映射；2972使用不同tile与record分解。完整CTA地址映射及其因果收益仍需配对实验，不能仅按WGM后缀归因。

## 与当前CK的受控PMC

使用DTK hipprof PMC type3另采100/911/1126，精确匹配赢家符号和实际CK路径。仅聚合最后一次完整G专家调用：matmul为最后G个kernel，grouped/CK为最后1个。初始化、参考计算、输入克隆等全部剔除。所有profile回放同样数值PASS。profile时延不用于性能排名。以下为报告计数，不是HBM字节或等待占比。

| 代表用例 | 后端 | 完整调用CTA数 | VALU | LDS | VMEM读 | LDS等待计数 |
|---|---|---:|---:|---:|---:|---:|
|100|BLAS|2816|22750208|3244032|1036288|2725550|
|100|CK|8192|29310976|5242880|1572864|578328|
|911|BLAS|768|31706112|6500352|1609728|9609104|
|911|CK|768|30950400|7077888|1572864|1831694|
|1126|BLAS|8192|497385472|117440512|29622272|178797050|
|1126|CK|5632|546308096|129499136|25231360|21451314|

100：BLAS完整G4调用2816个CTA，CK8192个。CK的LDS指令约多62%，VALU约多29%，而累计LDS等待更少。911：两者都是768个CTA，CK的LDS指令约多9%，而VALU、VMEM读、TCC请求及累计LDS等待更少。1126：CK的N384减少CTA和VMEM读，但VALU/LDS约多10%，仍略慢。三个代表均报告LDS bank conflict0、scratch0。因此不能把剩余问题简单定为带宽或LDS等待过高，也不能将静态低寄存器数作为获胜依据。

## 下一轮可检验实现

优先实现2972启发的内部M192/N256/K128、4搬运+8计算wave路径，显式处理内部M尾块（原始N2048不能整除192），保留原始NT接口及无scale INT32数学。先用原始权重作为内部A经LDS，原始激活作为直接B；不改变动态A，也不触碰16bit接口。独立Problem/Pipeline/Kernel/Epilogue，用本仓已有XOR LDS映射自行实现，不复制供应商指令块。比较完整A片段读取与既有小批读取，并在正确性证明后再对照4+4消费者交错；tile/操作数角色/同步的兼容组合应整体验证。

6950一组应以实际256×256/K128的交错调度和直接输入映射作对照，不能照名字改成K32。2979一组重点对齐两批消费者的B预取、完整A片段读取、record1024及CTA映射，再与已验证packed N384同轮比较。已有v65/v87等批读/早B负结果保留，不能把旧布局的失败直接推广到新M192映射。首先检查生成ISA是否保留预期重叠，再测原始G及全输出；只有稳定解决差距才接入维度驱动selector并新跑正式1178。

本轮没有推广新kernel，没有修改冻结Max目标，也没有更新正式全量CSV。当前正式状态1178数值PASS、1170达到目标、8条待解决；目标保持全部1178。


# Iteration117:independent BLAS-geometry-inspired CK prototypes

Implemented internalM192/N256/K128,4producer+8consumer waves,96INT32 accumulators/lane. Producer waves copy48 disjoint rows each(6vectors/lane);192x128payload occupies24KiB inside each32KiB ping-pong slot. Raw original weights are internalA/LDS,andraw activations internalB/record512-or1024 registers. Independent Problem/Pipeline/Kernel/Epilogue preserves original NT inputs. InternalM tail isexplicitly published withmasked source rows andguarded post-barrier outputs,including originalN2048's128-row finaltile. CPUbyte mapping PASS. Compare A4prefetch versus whole12-fragment A reads. Additional existing M256transpose/fullA16 variants compare4+4 consumers againstuniform8;they are controls,not aclaim ofinstruction-identical vendor scheduling.

Generated M192/A4 andA12 use169/185VGPR;M256split4/uniform8 use249/233VGPR. All64KiB LDS,zero spill/private.240full-rangeINT8/FP64 checks PASS withsingle/odd/even/longK,padding,small/partial internalM,ragged/empty diagnostic fallbacks,guards andimmutable inputs. Same-module rotated development screen covers20original shapes and80timed outputs,allnumericPASS.

No remaining target isclosed. M16384/N2048/K768/G4 current253.23TOPS;M192A4/A12 areabout11%slower. M256split4/fullA16 iscloser atabout1.4%slower. M2048/N3072/K4096/G8 current544.82TOPS;M192 variants about25-28%slower,M256split4 about15%slower. On1126,the control632.16TOPS islower thanv116's673.52;the apparent3.75%relative gainof45003 isnotaccepted andstilldoesnotreach the675.51target. Its cause isunproven;record this anomaly ratherthan changing formalCSV orselecting abetter-looking sample. No newfull1178run isjustified forthese defaults.

This tests internalorientation,geometry,A-read batching andexisting consumer split,butdoesnotyet reproduce allofBLAS2972'sphysicalLDS mapping orschedule. Infull vendorISA,the4producer waves interleave source rows and theirLDS blocks;consumer fragment bases/offset progression differ fromthe repository'scurrent XOR row mapping. M192prototype intentionally reuses our already-proven XOR mapping. Itsuniform8consumer schedule alsodiffers fromthe vendor's4+4interleaving. The M256control usesexisting CP code(priority3 onits legacy path),whereasvendorMMAC segmentsusepriority1. Thereforea negative result cannot beclaimed asproofthat the observed BLAS design cannot improveCK.

Next derive andvalidate the producer-to-LDS-to-MMAC mapping atbyte level withasm-independent CPUaudit andGPUstage readback,then make a compatible M1924+4stage schedule. Compare actual generated waits andload/MMAC ordering to thecomplete observed vendor loop. Include prologue/final odd/even stage,tail masks,andregisterpressure;do notcopy vendorinstruction blocks ormerelychange aTILEconstant. Reuse original8shapes andfixedMax targets;dimension-based production dispatch onlyafterstable benefit andindependent full acceptance. No FP16/BF16 interface changes,no dynamicA packing,noquantization.

Formal production remainsv112:1178numericalPASS,1170targets,8remaining. The prior116analysis/replay andthis experiment areseparate fromformal acceptance. No hardware-ceiling claim.


# Iteration118:GPU byte proof of producer/consumer LDS mapping

Diagnostic only,no production or timing claim. Built independent CK Problem/Pipeline/Kernel around a256-thread,4wave stage probe. Compare(0)existing XOR slab distribution,(1)interleaved source rows withoutm0 highbits,(2)the same rows withwave<<17 in m0. Initialize64KiB LDS witha sentinel,copy2stages ofraw row/K-identifiable INT32tags through16B VMEM-to-LDS,readbackthe complete LDS,andcaptureall12MIx2Khalvesx64lanesx4dword fragments perstage. Input isonlya byte-identifiable diagnostic,notnew GEMM arithmetic.

12profilesPASS:3modesx(valid rows192/128/32 withpitch256,androws192/pitch272). Eachobserved dword inall64KiB matches anindependent CPU model;allsource payload tags appear exactlyonce;output guards andcomplete inputs unchanged. For mode2,all12288fragment dwords perprofile exactlymatch logicalrow=MI*16+lane%16 andK=stage*128+half*64+(lane/16)*16+word*4. Mode0/1 pairedwiththemode2consumer addresses are deliberate negative layout controls,not numericalGEMM failures.

Derived map,forproducer wave w=0..3,lane l=0..63,vector i=0..5:

```
source_row = w + 4*(l/8) + 32*i
source_Kbyte = stage*128 + 16*(l%8)
LDS_byte = stage*32768 + i*4096 + w*1024 + ((16*l + 32*w) % 1024)
```

Thus m0'swave<<17 tag appliesa32*w-byte rotation inside each1KiB wave-copy region inthisverified gfx936/DTK mode. Do notgeneralize toother architectures/instructions. Thecopy sourceperwave interleavesrowsinstead ofour previouscontiguous48-row slabs. Payload remains24KiB per192x128stage insidea32KiB slot.

Anindependent consumer address avoids vendorlane-case masks. Forlogicalrow r,source16Bchunk c=0..7:

```
w=r%4; rg=(r%32)/4; panel=r/32
LDS_byte = stage*32768 + panel*4096 + w*1024
         + ((128*rg + 16*c + 32*w) % 1024)
```

ForMI=2q/parity andlane r0=lane%16,common=128*(r0/4)+16*(lane/16)+32*(r0%4). EvenMI usescommon(+64forhighKhalf);oddMI wrapscommon+512(+64high) within1KiB. Addq*4096andwavebase1024*(r0%4). CompleteGPUreadbackproves these addresses,including thewrapping lanes. Initialtranscription missed twohigh-half/oddMI boundary lanes45/61;itproduced96badfragmentdwords ona full192-rowprofilewhilepayloadwascorrect. Corrected via theindependent map,andkept originalsummary/source/binary underresults/lds_map_v118 andbuild/v118/initial. The final mode2has0bad fragmentsinall4profiles.

Firstcompiler attemptallocatedaVGPR tom0'ssource. Explicitreadfirstlane onthewave-uniform destination producesa scalaroperand. Thiswasacompilation issue,notGPUavailability. FinalbinarySHA andprofiledetails areintheJSON. No rawvendorinstructionblock iscommitted;formulasandprobeareowncode.

Next integrate the proven interleaved/rotated publication andfragmentbases intoM192transposed GEMM,thenvalidate4+4staggered consumers andpriority1 independently. PriorM192prototype'snegative result didnotinclude thismapping. RequirefullrangeINT8,tail/odd/evenstagechecks andsame-moduletiming beforepromotion. Formalv112remains1178numericPASS/1170targets/8gaps;FP16/BF16interfacesunchanged.
