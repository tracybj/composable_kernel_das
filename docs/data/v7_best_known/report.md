# 全1193 shape：当前已测最优方案汇总

生成时间：2026-11-05T10:00:22.265106+08:00（北京时间）。本次只做CPU离线汇总，没有运行GPU测试，没有更改kernel、selector或默认开关。

每个shape一行，确实从已测记录中取数值最小的方案；不推测未测组合或未来优化。这里的“最优”是已测集合中的历史观测最优，不代表已经证明它在所有运行时段都最快。

## 先看哪张表

- `best_known_1193.csv`：主表，全部历史已测方案的观测最优，包含历史M01=4等已实测、当前需恢复源码的配置。
- `best_current_1193.csv`：只在当前代码已有开关可选中的方案中取最优，仍是历史已测数值；全部1193条都有结果。
- `best_typical_1193.csv`：辅助稳健视图。先对同一方案各实验系列的统计值再取中位数，然后比较方案，减少只取某个时段最低值的影响。
- `all_candidate_series.csv` / `all_candidate_series.jsonl.gz`：所有方案的实验系列统计值；`all_observations.jsonl.gz`保留每一条原始测量的数值、来源键和哈希。

共读取296个来源文件，保留14,250条有效观测，其中主比较为12,875条同口径CK观测，形成5,994个方案/实验系列统计值，覆盖1193条：881 decode、312 prefill。

## 比较规则

主表统一为物理卡4、device_args benchmark、BF16 NN、common非ragged/empty、单launch、warmup10/samples30、include-args-build=0。每条测量本来就是30个样本的中位数；三轮实验再取三轮中位数，六轮则保留六轮。单轮全量快照只有一个30样本中位数，会标记缺少三轮确认。绝不从样本min_ms、最快一轮中取值。

同一个方案在不同实验/对照系列里有多个统计值时，主表保留最低的那个并注明来源；因此它是跨时段的历史观测包络，不是新的统一全量测试。辅助typical表则使用同一方案各系列统计值的中位数。两种汇总有35条shape的方案名称不同。

卡0的CK实验、fused_test连续20次launch数据保留在other_protocol_reference.csv，不参与主表选优。BLAS仍使用冻结的旧1178条卡0、新增15条卡4的Native repeat20数据，未重新调优、未选最快BLAS。R_frozen=冻结BLAS/单launch CK，涉及跨协议及部分跨卡，只作历史参考。已知BLAS采样跨度大于50%的行另有BLAS_high_span_warning标记。

更早的results_bfnn.csv没有记录CK的GPU、计时方式、实例及精度状态，其1178条数值保留在unknown_protocol_ck_reference.csv，未混入主比较。派生comparison.csv不作为独立测量；P4a timing.txt与timing_6.txt逐字节相同，仅算一次。

P6e已明确排除的33个环境异常键全部排除并保留在excluded_records.jsonl.gz；失败运行、精度失败、附录之外的MNKG也不参与。11-03全量历史存在MKFD/服务恢复流程较弱的记录，未伪装成新鲜健康测量；若其数值胜出则标记legacy_environment_warning。

实际按MNKG匹配shape，不单看日志文件名的ID。P3a有G16代理测量，15条代理观测按实际G16及MNK归入对应shape，未外推到原G32/G64等shape。映射完整保存在shape_label_remaps.json。

## 观测最优方案分布

|方案|数值最低的shape条数|
|---|---:|
|V7原几何，M01=1|462|
|既有小M N128/K16|332|
|NoSLC，M01=8|189|
|小M N128/K16 drain|118|
|四wave 128×128/K16|22|
|小M N128/K32|22|
|N1232原回退4003|16|
|V7原几何，M01=8|16|
|NoSLC，M01=1|7|
|历史M01=4实验|5|
|历史P2实现|3|
|历史P1实现|1|

其中10条的历史最优不能通过当前开关直接启用，主表明确标出current_switch_available=False，并同时列出best_current_scheme/best_current_ms。历史配置包括5条M01=4、4条P1/P2旧实现，以及1条在当前K阈值之外实测的M01=8。

drain或square整体实验有回退，仍可能在个别shape成为数值最优；这次用户明确要求逐shape已测上限，所以这些行照实保留。此前实验的完整回退表和验收结论没有改动；本表不代表已实现按ID挑核的生产策略。

## 代表shape

|ID|当前已测数值最优|CK ms|冻结BLAS ms|R_frozen|实验系列|
|---|---|---:|---:|---:|---|
|1|小M N128/K16 drain|0.124640|0.141680|1.137|p6e/smallperf_drain_after|
|33|四wave 128×128/K16|0.031200|0.024848|0.796|p6e/smallperf_square_after|
|76|小M N128/K32|0.021600|0.017048|0.789|p6e/smallexpand_k32_lowcta_after|
|84|四wave 128×128/K16|0.034720|0.026656|0.768|p6e/smallperf_square_after|
|146|N1232原回退4003|5.266028|0.821168|0.156|full1178|
|358|既有小M N128/K16|0.637044|0.670040|1.052|p6c/perf_n128|
|633|小M N128/K32|0.036640|0.032816|0.896|p6e/smallperf_k32_after|
|639|四wave 128×128/K16|0.062880|0.054144|0.861|p6e/smallperf_square_after|
|681|小M N128/K32|0.033759|0.027664|0.819|p6e/smallexpand_k32_lowcta_after|
|689|四wave 128×128/K16|0.055041|0.047384|0.861|p6e/smallperf_square_after|
|1034|历史M01=4实验|15.803927|16.142246|1.021|P4a_order6/all/m01_4|
|1038|历史M01=4实验|15.778181|16.467299|1.044|P4a_order_wide/all/m01_4|
|1177|NoSLC，M01=8|59.501564|58.563709|0.984|p6e/expand_cache_noslc|
|1178|NoSLC，M01=8|29.820374|29.023935|0.973|p6e/expand_cache_noslc|
|1186|四wave 128×128/K16|0.042159|0.038440|0.912|p6e/smallexpand_square_m128_256_after|
|1187|NoSLC，M01=1|0.057600|0.050400|0.875|p6e/expand_cache_noslc|
|1193|NoSLC，M01=1|1.734480|1.768687|1.020|p6e/expand_cache_noslc|

## 全1193条的参考分桶

以下是历史逐shape观测最优拼接的分桶，不是某个已实现策略的全量实测成绩，尤其不能把跨卡、repeat20/单launch的R当成同口径胜负。

|phase|总数|R≥.99|[.8,.99)|[.5,.8)|[.3,.5)|[.2,.3)|<.2|
|---|---:|---:|---:|---:|---:|---:|---:|
|prefill|312|287|21|4|0|0|0|
|decode|881|594|257|14|7|8|1|
|all|1193|881|278|18|7|8|1|

## 如何读质量标记

主表始终写出实际数值最低的方案，没有因为差距小而偷偷换成别的方案。quality、next_scheme、gap_to_next_scheme_pct、within2pct_schemes用于区分数值最低与明确领先。标记可以叠加，数量不能相加当作总条数。

|标记|含义|条数|
|---|---|---:|
|clear_numerical_lead_low_reported_variation_not_paired_proof|数值差距及已记录波动较清楚，仍非跨时段配对因果证明|318|
|gap_within_reported_variation|差距未超过2%、两倍样本CV、轮间跨度的保守观察阈值|500|
|high_variation|样本CV>5%或系列轮间跨度>5%|82|
|historical_source_variant_not_current_switch|历史源码/配置，不能由当前开关直接选中|10|
|legacy_environment_warning|最低值来自11-03历史环境存在波动的快照|123|
|near_tie_within2pct|与次优不同方案差距≤2%，近似并列|397|
|only_one_scheme_measured|只有一种方案的可比测量，不表示排除了其他未测方案|252|
|runner_sample_CV_not_fully_recorded|次优记录没有完整保留样本CV|23|
|sample_CV_not_fully_recorded|旧记录没有完整保留样本CV|9|
|single_series_no3round_confirmation|该最低系列不足三轮|419|

这些是描述性质量标记，不是统计置信区间。未测的K32+drain等组合没有填入；理论上的未来改进不参与。

## 列说明和复核

- best_scheme / best_instance_id / best_ms：已测最优方案、当前对应ID及该实验系列统计耗时；历史M01=4没有当前ID，recorded_instance_ids仍保留当时实际ID5000。
- best_candidate_id：可在all_candidate_series.csv定位完整轮次与observation_ids；每条observation再定位原文件及source_key。
- best_source / best_source_keys / best_binary_sha256：原始数据文件、行号或job键及已记录的二进制SHA。旧资料未保存的binary SHA留空，不推造。source_catalog.json记录所有使用文件的SHA256。
- current_switch_available / enable_env：当前代码能否由开关选中及开关参数；实验开关需在清洁环境下使用，其他实验开关不设。只有CPU selector可用性检查，不代表重新测过性能或精度。
- best_current_scheme / best_current_ms：排除当前无法直接启用的历史源码配置后，已有记录中的最优值。
- typical_best_scheme / typical_best_ms：对每个方案的各实验系列统计值再取中位数后的最优方案，用来观察跨系列稳定性。
- original_reference_ms / observed_delta_vs_reference_pct：以P6c原CK对照列为参考的历史数值差，非新鲜配对优化收益。

验证通过：全部1193条ID/MNKG、三轮统计值复算、各行最小值、33个强制排除键、296个来源文件哈希及分桶均已复核。当前可用方案共2663个(shape,scheme)组合通过纯CPU selector匹配，覆盖1193条。

复现（全部CPU离线）：

```bash
python3 /workspace/v7_coverage/best_known/build.py
python3 /workspace/v7_coverage/best_known/selector_audit.py
python3 /workspace/v7_coverage/best_known/validate.py
python3 /workspace/v7_coverage/best_known/report.py
```

没有改动原实验表、冻结BLAS、1193输入、任何kernel/selector、实验默认状态或原readfirstlane未提交修改。只新增汇总、审计和本地数据提交。
