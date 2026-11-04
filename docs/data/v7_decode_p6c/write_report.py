#!/usr/bin/env python3
"""Human-readable comparison with explicit measured/carried data provenance."""
import csv
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent

def main():
 tables=json.loads((ROOT/'tables.json').read_text())
 measured=json.loads((ROOT/'measured.json').read_text())
 checks=json.loads((ROOT/'validation.json').read_text())
 merged=list(csv.DictReader((ROOT/'merged1193.csv').open()))
 t={(r['dataset'],r['phase']):r for r in tables}
 applied=[r for r in measured if r['apply_policy']]
 lines=['# P6c：小 M 原型的增量分桶评估（2026-11-04）','',
  '**本报告是增量实验拼接结果。** CK实测615个不同shape：原600条优化候选，加新增15条；原578条沿用11-03。主方案仅将M≤64的607条切至5003/Tn128；其余保持原路径。统计策略由测试范围和开关设置模拟，当前源码的实验入口仍支持M≤256，生产默认保持原路径。没有运行1193条全量，没有改kernel。','',
  f"预先固定策略范围内，三轮A/B耗时变化中位数为 **{checks['median_delta_pct']:+.2f}%**；改善超过2%的{checks['improved_gt2']}条，回退超过2%的{checks['regressions_gt2']}条、超过5%的{checks['regressions_gt5']}条。全部测量值均保留，不按每条实测选快者。",'',
  f"这607个shape各调用一次的kernel耗时总和：{checks['one_call_before_ms']:.6f}→{checks['one_call_after_ms']:.6f} ms（{checks['one_call_delta_pct']:+.2f}%）；逐shape耗时比的几何平均变化{checks['geomean_delta_pct']:+.2f}%。总和由大权重shape主导；它和尾部case数量衡量的是不同方面。",'',
  'R=BLAS ms/CK ms（等价于相同shape的CK吞吐/BLAS吞吐）。互斥区间与截图一致，各列之和等于total-case。原shape的BLAS来自卡0冻结GroupedGemm，CK卡4；新增BLAS和CK均在卡4。新增BLAS采用与冻结基线相同的20次launch协议，CK仍为单launch，两边协议差异沿用历史口径。','',
  '## 原1178条：复现截图及本次增量对照','',
  '“本次对照”以本次600条原路径A/B结果替换其历史时间，其余578条沿用11-03；“本次优化”在相同范围替换为5003。两者比较用于区分优化收益与历史测量时段漂移。10-30/11-03两次旧结果按原值复现。','',
  '| 数据 | phase | total-case | [0.99,∞) | [0.8,0.99) | [0.5,0.8) | [0.3,0.5) | [0.2,0.3) | <0.2 |',
  '|---|---|---|---|---|---|---|---|---|']
 names=[('10-30','10-30'),('11-03','11-03'),('current_control_original1178','本次对照（增量）'),('current_optimized_original1178','本次优化（增量）')]
 def row(label,r):return '| '+label+' | '+r['phase']+' | '+str(r['total_case'])+' | '+' | '.join(map(str,r['buckets']))+' |'
 for key,label in names:
  for phase in ['prefill','decode']:lines.append(row(label,t[key,phase]))
 b=t['current_control_original1178','decode'];a=t['current_optimized_original1178','decode']
 lines+=['',f"相同复测范围的decode：R<0.8从{b['R_lt_08']}变为{a['R_lt_08']}；R<0.5从{b['R_lt_05']}变为{a['R_lt_05']}；R≥0.99从{b['buckets'][0]}变为{a['buckets'][0]}。prefill原306条均沿用历史结果。",'',
  '## 加入新增15条：扩展1193条参考表','',
  '新增9条decode（1179…1187）和6条prefill（1188…1193）。其中M≤64的7条应用主方案；新增M128/256仅做额外原型对照，主方案保留其原路径。此表含原BLAS卡0与新增BLAS卡4两种来源，属于扩展参考分布；原1178的同口径结论应看上一表。','',
  '| 数据 | phase | total-case | [0.99,∞) | [0.8,0.99) | [0.5,0.8) | [0.3,0.5) | [0.2,0.3) | <0.2 |',
  '|---|---|---|---|---|---|---|---|---|']
 for key,label in [('current_control_extended1193','本次对照（增量）'),('current_optimized_extended1193','本次优化（增量）')]:
  for phase in ['prefill','decode']:lines.append(row(label,t[key,phase]))
 lines+=['','## 新增shape逐条结果','',
  '时间均为卡4 ms。旧CK为N128配对中的三轮中位数；N64的变化使用其独立before→after基线，可能与表中旧CK略有差异。两种tile的数值是实验分支结果，不代表主方案全部采用。M≥512仅测旧路径。BLAS为新native GroupedGemm基线。','',
  '| ID | M | phase | 旧CK | Tn128 | 变化 | Tn64 | 变化 | BLAS | 主方案R |',
  '|---|---|---|---|---|---|---|---|---|---|']
 for c in merged:
  i=int(c['ID'])
  if i<1179:continue
  bytile={r['tile_n']:r for r in measured if r['ID']==i}
  r128=bytile.get(128);r64=bytile.get(64)
  vals=[str(i),c['M'],c['phase'],f"{float(c['CK_before_ms']):.6f}"]
  for r in [r128,r64]:vals+= [f"{r['after_ms']:.6f}",f"{r['delta_pct']:+.1f}%"] if r else ['—','—']
  vals += [f"{float(c['BLAS_ms']):.6f}",f"{float(c['R_after']):.3f}"]
  lines.append('| '+' | '.join(vals)+' |')
 lines+=['','## 回退与边界','',
  '旧R<0.8的149条中，主方案覆盖113条，其余36条保留原路径。旧R<0.5的36条中，覆盖20条、保留16条；保留的16条全部为N=1232。因此即使候选全部改善，也不应预期本轮清空最差桶。','',
  '以下按主方案的三轮耗时变化排序，展示最慢的15条；不删除回退行，也不将未验证的按shape保底选择计入主结果。具体轮间值/CV保存在measured.json，全部615条来源见merged1193.csv。','',
  '| ID | M,N,K,G | 旧CK卡4 ms | 新CK卡4 ms | 变化 | R前→后 |',
  '|---|---|---|---|---|---|']
 for r in sorted(applied,key=lambda r:-r['delta_pct'])[:15]:
  lines.append(f"| {r['ID']} | {r['M']},{r['N']},{r['K']},{r['G']} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['delta_pct']:+.1f}% | {r['R_before']:.3f}→{r['R_after']:.3f} |")
 consistent=sum(all(a/b>1.02 for a,b in zip(r['after_round_ms'],r['before_round_ms'])) for r in applied)
 lines+=['',f'三轮各自都回退>2%的{consistent}条；三轮中位数回退>2%的{checks["regressions_gt2"]}条。小kernel阈值附近的分桶受计时噪声影响，全部原值保留。主策略仍为实验，不能以部分收益宣称全部小M无回退。','',
  '## 原600条按属性分段','',
  '这里仅包含原600条实测候选，BLAS均为卡0冻结值。CTA按原5000的256×256 tile计算。分段用于解释适用范围，未据此逐条挑选结果替换主表。','',
  '| 分组 | 条数 | 耗时变化中位数 | 回退>2% | R<0.8前→后 |','|---|---|---|---|---|']
 old_applied=[r for r in applied if r['ID']<=1178]
 groups=[]
 for lo,hi in [(0,40),(40,80),(80,160),(160,320),(320,640),(640,10**12)]:
  group=[r for r in old_applied if lo<=((r['M']+255)//256)*((r['N']+255)//256)*r['G']<hi]
  groups.append((f'CTA {lo}…{hi-1}' if hi<10**12 else 'CTA≥640',group))
 for lo,hi,label in [(64,1024,'K<1024'),(1024,2048,'1024≤K<2048'),(2048,4096,'2048≤K<4096'),(4096,4097,'K=4096')]:
  groups.append((label,[r for r in old_applied if lo<=r['K']<hi]))
 for label,group in groups:
  if not group:continue
  lines.append(f"| {label} | {len(group)} | {st.median(r['delta_pct'] for r in group):+.2f}% | {sum(r['delta_pct']>2 for r in group)} | {sum(r['R_before']<.8 for r in group)}→{sum(r['R_after']<.8 for r in group)} |")
 lines+=['','## 验证、命令与来源','',
  f"`INCREMENTAL_CAMPAIGN_PASS valid_jobs={checks['valid_jobs']}`：accuracy18、newblas15、perf3726。新增decode18组分支检查，每组3次，计{checks['full_output_comparisons']}次BLAS全输出比对、{checks['full_output_elements']}个元素，failed=0 max_abs=0。全部CK计时执行CPU抽样及guard检查；未将600条抽样称为全输出检查。此前同一二进制P6b的162次全输出验证见P6b报告。",'',
  f"另保留{checks['environment_failed_attempts']}次环境启动失败（总尝试{checks['total_attempts']}）。第三轮旧基线ID759在SELECT之前报invalid device ordinal，未产生精度/性能结果；同一时段内核日志出现hymgr注销和反复注册。待卡4恢复、最近事件超过两分钟后原地重试成功。仅无结果的启动失败不计入性能样本，所有已产生的性能值均保留；详见environment_failure.json及完整jobs日志。",'',
  'native GroupedGemm新基线15条均通过现有工具输入完整性、输出guard和全输出参考校验；候选heuristic64、workspace32MiB、screen warmup3/repeat10/samples3、top3，正式warmup5/repeat20/samples3。原1178 BLAS不重新调优。','',
  '所有GPU命令物理卡4串行，逐条hy-smi确认HCU%=0；MKFD/缺行最多3次60秒重试，忙卡立即停止。正式CK走device_args_benchmark与默认selector；实验开关按预先固定范围设置，未强制expected ID。warmup10、samples30、每sample单launch、include-args-build=0。原5000/5001/5002和readfirstlane保护状态保持。','',
  '```bash','ulimit -c 0',
  'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6c/run.py accuracy',
  'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6c/run.py newblas',
  'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6c/run.py perf',
  'python3 /workspace/v7_coverage/p6c/analyze.py',
  'python3 /workspace/v7_coverage/p6c/write_report.py','```','',
  'plan.json固定实验范围；manifest.json/script_sha256.json记录输入、二进制、依赖和脚本；jobs.jsonl/gate.jsonl逐条记录命令、日志位置、门禁和结果。原始日志和新增BLAS候选数据留在 /workspace/v7_coverage/p6c/，不放入Git。measured.csv/JSON是实测，merged1193.csv逐条标明实测与沿用来源。','',
  '本轮原600条的75条回退全部出现在K≥2048。M≤64且K<2048的原182条及新增7条（共189条）没有>2%回退，可作为下一阶段收窄入口的观察依据；K=2048的低CTA尾部仍应单独保留重点评估。这是根据本轮数据提出的后续建议，未用该子集改写主表。','',
  '本阶段仅输出增量评估，默认选核仍未提升。M128/256、N尾块1232、短K浅流水不是本次实现范围；下一阶段如何收窄属性范围，应基于完整回退数据决定。','']
 (ROOT/'report.md').write_text('\n'.join(lines))
 print('REPORT',ROOT/'report.md')

if __name__=='__main__':main()
