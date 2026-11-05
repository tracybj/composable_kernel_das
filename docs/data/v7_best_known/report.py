#!/usr/bin/env python3
"""Human-readable map to the historical measured envelope and current-switch view."""
import csv,json
from pathlib import Path

P=Path(__file__).resolve().parent
NAMES={
 'v7_256x256x16_M01_1':'V7原几何，M01=1',
 'v7_256x256x16_M01_8':'V7原几何，M01=8',
 'smallM_64x128x16':'既有小M N128/K16',
 'smallM_64x128x16_drain':'小M N128/K16 drain',
 'smallM_64x128x32':'小M N128/K32',
 'smallM_128x128x16_square':'四wave 128×128/K16',
 'v7_256x256x16_NoSLC_M01_1':'NoSLC，M01=1',
 'v7_256x256x16_NoSLC_M01_8':'NoSLC，M01=8',
 'v4_fallback_N_tail':'N1232原回退4003',
 'historical_v7_M01_4':'历史M01=4实验',
 'legacy_P1_v7_256x256x16_M01_1':'历史P1实现',
 'legacy_P2_v7_256x256x16_M01_1':'历史P2实现'}

def main():
 s=json.loads((P/'summary.json').read_text());audit=json.loads((P/'validation.json').read_text());assert audit['passed']
 rows=json.loads((P/'best_known_1193.json').read_text());index={row['ID']:row for row in rows}
 lines=['# 全1193 shape：当前已测最优方案汇总','',f'生成时间：{s["created_at"]}（北京时间）。本次只做CPU离线汇总，没有运行GPU测试，没有更改kernel、selector或默认开关。','',
 '每个shape一行，确实从已测记录中取数值最小的方案；不推测未测组合或未来优化。这里的“最优”是已测集合中的历史观测最优，不代表已经证明它在所有运行时段都最快。','',
 '## 先看哪张表','',
 '- `best_known_1193.csv`：主表，全部历史已测方案的观测最优，包含历史M01=4等已实测、当前需恢复源码的配置。',
 '- `best_current_1193.csv`：只在当前代码已有开关可选中的方案中取最优，仍是历史已测数值；全部1193条都有结果。',
 '- `best_typical_1193.csv`：辅助稳健视图。先对同一方案各实验系列的统计值再取中位数，然后比较方案，减少只取某个时段最低值的影响。',
 '- `all_candidate_series.csv` / `all_candidate_series.jsonl.gz`：所有方案的实验系列统计值；`all_observations.jsonl.gz`保留每一条原始测量的数值、来源键和哈希。','',
 f'共读取{s["source_files"]}个来源文件，保留{s["observations"]:,}条有效观测，其中主比较为{s["primary_observations"]:,}条同口径CK观测，形成{s["primary_candidate_series"]:,}个方案/实验系列统计值，覆盖1193条：881 decode、312 prefill。','',
 '## 比较规则','',
 '主表统一为物理卡4、device_args benchmark、BF16 NN、common非ragged/empty、单launch、warmup10/samples30、include-args-build=0。每条测量本来就是30个样本的中位数；三轮实验再取三轮中位数，六轮则保留六轮。单轮全量快照只有一个30样本中位数，会标记缺少三轮确认。绝不从样本min_ms、最快一轮中取值。','',
 '同一个方案在不同实验/对照系列里有多个统计值时，主表保留最低的那个并注明来源；因此它是跨时段的历史观测包络，不是新的统一全量测试。辅助typical表则使用同一方案各系列统计值的中位数。两种汇总有'+str(s['observed_vs_typical_scheme_disagreements'])+'条shape的方案名称不同。','',
 '卡0的CK实验、fused_test连续20次launch数据保留在other_protocol_reference.csv，不参与主表选优。BLAS仍使用冻结的旧1178条卡0、新增15条卡4的Native repeat20数据，未重新调优、未选最快BLAS。R_frozen=冻结BLAS/单launch CK，涉及跨协议及部分跨卡，只作历史参考。已知BLAS采样跨度大于50%的行另有BLAS_high_span_warning标记。','',
 '更早的results_bfnn.csv没有记录CK的GPU、计时方式、实例及精度状态，其1178条数值保留在unknown_protocol_ck_reference.csv，未混入主比较。派生comparison.csv不作为独立测量；P4a timing.txt与timing_6.txt逐字节相同，仅算一次。','',
 'P6e已明确排除的33个环境异常键全部排除并保留在excluded_records.jsonl.gz；失败运行、精度失败、附录之外的MNKG也不参与。11-03全量历史存在MKFD/服务恢复流程较弱的记录，未伪装成新鲜健康测量；若其数值胜出则标记legacy_environment_warning。','',
 '实际按MNKG匹配shape，不单看日志文件名的ID。P3a有G16代理测量，15条代理观测按实际G16及MNK归入对应shape，未外推到原G32/G64等shape。映射完整保存在shape_label_remaps.json。','',
 '## 观测最优方案分布','',
 '|方案|数值最低的shape条数|','|---|---:|']
 for name,count in sorted(s['winners_by_scheme'].items(),key=lambda item:-item[1]):lines.append(f'|{NAMES.get(name,name)}|{count}|')
 lines+=['',f'其中{s["historical_winners_unavailable"]}条的历史最优不能通过当前开关直接启用，主表明确标出current_switch_available=False，并同时列出best_current_scheme/best_current_ms。历史配置包括5条M01=4、4条P1/P2旧实现，以及1条在当前K阈值之外实测的M01=8。','',
 'drain或square整体实验有回退，仍可能在个别shape成为数值最优；这次用户明确要求逐shape已测上限，所以这些行照实保留。此前实验的完整回退表和验收结论没有改动；本表不代表已实现按ID挑核的生产策略。','',
 '## 代表shape','',
 '|ID|当前已测数值最优|CK ms|冻结BLAS ms|R_frozen|实验系列|','|---|---|---:|---:|---:|---|']
 for i in [1,33,76,84,146,358,633,639,681,689,1034,1038,1177,1178,1186,1187,1193]:
  row=index[i];lines.append(f'|{i}|{NAMES.get(row["best_scheme"],row["best_scheme"])}|{row["best_ms"]:.6f}|{row["BLAS_ms"]:.6f}|{row["R_frozen"]:.3f}|{row["best_campaign"]}|')
 lines+=['','## 全1193条的参考分桶','',
 '以下是历史逐shape观测最优拼接的分桶，不是某个已实现策略的全量实测成绩，尤其不能把跨卡、repeat20/单launch的R当成同口径胜负。','',
 '|phase|总数|R≥.99|[.8,.99)|[.5,.8)|[.3,.5)|[.2,.3)|<.2|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in s['buckets']:lines.append('|'+row['phase']+'|'+'|'.join(str(row[key]) for key in ['total','R_ge099','R_08_099','R_05_08','R_03_05','R_02_03','R_lt02'])+'|')
 lines+=['','## 如何读质量标记','',
 '主表始终写出实际数值最低的方案，没有因为差距小而偷偷换成别的方案。quality、next_scheme、gap_to_next_scheme_pct、within2pct_schemes用于区分数值最低与明确领先。标记可以叠加，数量不能相加当作总条数。','',
 '|标记|含义|条数|','|---|---|---:|']
 descriptions={
 'near_tie_within2pct':'与次优不同方案差距≤2%，近似并列',
 'gap_within_reported_variation':'差距未超过2%、两倍样本CV、轮间跨度的保守观察阈值',
 'high_variation':'样本CV>5%或系列轮间跨度>5%',
 'only_one_scheme_measured':'只有一种方案的可比测量，不表示排除了其他未测方案',
 'single_series_no3round_confirmation':'该最低系列不足三轮',
 'legacy_environment_warning':'最低值来自11-03历史环境存在波动的快照',
 'sample_CV_not_fully_recorded':'旧记录没有完整保留样本CV',
 'runner_sample_CV_not_fully_recorded':'次优记录没有完整保留样本CV',
 'historical_source_variant_not_current_switch':'历史源码/配置，不能由当前开关直接选中',
 'clear_numerical_lead_low_reported_variation_not_paired_proof':'数值差距及已记录波动较清楚，仍非跨时段配对因果证明'}
 for flag,count in sorted(s['winner_quality_counts'].items()):lines.append(f'|{flag}|{descriptions.get(flag,flag)}|{count}|')
 lines+=['','这些是描述性质量标记，不是统计置信区间。未测的K32+drain等组合没有填入；理论上的未来改进不参与。','',
 '## 列说明和复核','',
 '- best_scheme / best_instance_id / best_ms：已测最优方案、当前对应ID及该实验系列统计耗时；历史M01=4没有当前ID，recorded_instance_ids仍保留当时实际ID5000。',
 '- best_candidate_id：可在all_candidate_series.csv定位完整轮次与observation_ids；每条observation再定位原文件及source_key。',
 '- best_source / best_source_keys / best_binary_sha256：原始数据文件、行号或job键及已记录的二进制SHA。旧资料未保存的binary SHA留空，不推造。source_catalog.json记录所有使用文件的SHA256。',
 '- current_switch_available / enable_env：当前代码能否由开关选中及开关参数；实验开关需在清洁环境下使用，其他实验开关不设。只有CPU selector可用性检查，不代表重新测过性能或精度。',
 '- best_current_scheme / best_current_ms：排除当前无法直接启用的历史源码配置后，已有记录中的最优值。',
 '- typical_best_scheme / typical_best_ms：对每个方案的各实验系列统计值再取中位数后的最优方案，用来观察跨系列稳定性。',
 '- original_reference_ms / observed_delta_vs_reference_pct：以P6c原CK对照列为参考的历史数值差，非新鲜配对优化收益。','',
 f'验证通过：全部1193条ID/MNKG、三轮统计值复算、各行最小值、33个强制排除键、{audit["source_files_verified"]}个来源文件哈希及分桶均已复核。当前可用方案共{audit["selector_cases"]}个(shape,scheme)组合通过纯CPU selector匹配，覆盖1193条。','',
 '复现（全部CPU离线）：','',
 '```bash\npython3 /workspace/v7_coverage/best_known/build.py\npython3 /workspace/v7_coverage/best_known/selector_audit.py\npython3 /workspace/v7_coverage/best_known/validate.py\npython3 /workspace/v7_coverage/best_known/report.py\n```','',
 '没有改动原实验表、冻结BLAS、1193输入、任何kernel/selector、实验默认状态或原readfirstlane未提交修改。只新增汇总、审计和本地数据提交。','']
 (P/'report.md').write_text('\n'.join(lines));print('BEST_KNOWN_REPORT_WRITTEN')

if __name__=='__main__':main()
