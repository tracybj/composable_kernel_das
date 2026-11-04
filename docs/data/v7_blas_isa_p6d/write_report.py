#!/usr/bin/env python3
"""Build the reviewable phase report from all successful rounds."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
 ts=json.loads((ROOT/'timing.json').read_text())
 ps=json.loads((ROOT/'profiles.json').read_text())
 vs=json.loads((ROOT/'validation.json').read_text())
 features=json.loads((ROOT/'isa_features.json').read_text())
 coverage=json.loads((ROOT/'coverage.json').read_text())
 assert vs['timing_jobs']==75 and vs['profile_jobs']==13
 out=['# P6d：BLAS反汇编与CK迁移方向诊断（实验用，不合入）','',
 '本阶段完成实际Native GroupedGemm内核反汇编、九条代表shape三轮同卡复测及13组profiler。包含新增ID1186/1187和大M大K ID1178；没有跑1193全量，没有修改CK kernel、selector、默认开关或N尾块契约。目标仍为全部1193条接近/超过BLAS，沿用R=BLAS/CK及0.99分桶；本阶段没有实现这一目标。','',
 '## 1. 范围、依赖与计时口径','',
 '- 实际BLAS库与gfx936 code object见manifest；不是泛指另一版rocBLAS或从网上推测。使用已安装DTK llvm-objdump与用户提供的xprof/XCompute说明；未安装Windows GUI。冻结算法ID与kernel_name逐项断言，不重调BLAS。',
 '- BLAS桥接以列主序表示C^T=B^T×A^T，Native M=N、N=M；以下tile全部转回CK逻辑M×N×K。',
 '- 6个BLAS kernel、5个CK V7 kernel全部正常解码（unknown=0）；选定BLAS内核覆盖历史映射724/1193条、6/14个名称，**724不是GPU复测条数**。CK静态LDS=0表示使用动态LDS，不能读作“不使用LDS”。',
 '- 全部GPU命令固定物理卡4，串行且逐条hy-smi空闲门禁。CK默认selector，关闭/打开既有small-M128实验开关，warmup10/samples30/单launch/include-args-build0；Native同进程分别测repeat1、repeat20，各30样本。三轮顺序为CK原路径→small128→BLAS。',
 '- 新鲜同卡、同单launch比较用于内核差距诊断；历史R保留卡0 BLAS/卡4 CK口径，新增BLAS为卡4。Native repeat20会改变短kernel的事件/提交摊销与连续运行状态，不能把它与CK单launch的差距全部归因于shader；也不能用同口径诊断表静默替换历史分桶。',
 '- 计时取三轮中位数，timing.json保留每轮中位数/CV和所有样本范围，未挑最快轮。profiler耗时包含重放和频率假设，仅用于诊断，不替代正式时间。','',
 '## 2. 三轮代表性结果（全部物理卡4，ms）','',
 '| ID | M/N/K/G | 原CK | 已有5003 | Native单launch | Native连续20次/次 | 5003相对原CK |',
 '|---|---|---:|---:|---:|---:|---:|']
 for r in ts:
  val=lambda b:f"{r[b]['median_ms']:.6f}" if b in r else '—'
  delta=f"{r['small_delta_pct']:+.2f}%" if 'small_delta_pct' in r else '—'
  out.append(f"|{r['id']}|{r['M']}/{r['N']}/{r['K']}/{r['G']}|{val('base')}|{val('small128')}|{val('blas')}|{val('blas_repeat20')}|{delta}|")
 out += ['',
 '5003一列是既有原型的对照，不是本阶段写出的新优化。原路径ID146为4003，ID1178为5002，其余5000；small128均默认selector选到5003。N1232与大M未强行放入小M入口。','',
 '## 3. 已验证的ISA差异','',
 '| BLAS对应ID / CK实例 | 逻辑tile | 线程 | VGPR/SGPR | LDS/CTA |',
 '|---|---|---:|---:|---:|',
 '| BLAS 76/633 | 64×64×32 | 256 | 102/100 | 32KB |',
 '| BLAS 681 | 64×128×32 | 256 | 186/100 | 64KB |',
 '| BLAS 358、772（两种WGM） | 128×128×16 | 256 | 186/100 | 32KB |',
 '| BLAS 146/1186/1187 | 128×128×16 | 256 | 256/76 | 32KB |',
 '| BLAS 1178 | 256×256×16 | 512 | 236/100 | 64KB |',
 '| CK 5000/5002 | 256×256×16 | 512 | 212/47 | 动态64KB |',
 '| CK 5003 | 64×128×16 | 256 | 80/47 | 动态32KB |',
 '| CK 5004 | 64×64×16 | 256 | 55/51 | 动态32KB |','',
 '**K32是真实的流水级，不只是名字。** 76/633/681的K循环计数来自K>>5；128平方和256平方来自K>>4。681每级3次Direct-to-LDS、6次LDS读取、16条MMAC，CK5003每级2次Direct-to-LDS、4次LDS读取、8条MMAC。两者相同逻辑64×128 tile，K2048时矩阵指令总量相同；BLAS用更大的K级、不同operand组织和更长的MMAC burst。仅把KPerBlock模板数字改32不会自动重建这些映射。','',
 '**6个BLAS内核都有停止全局加载的K收尾。** 第一个oddexit_31块的buffer_load均为0，但仍分别执行32/64/64/64/64/128条MMAC，按剩余路径消化已在LDS的数据；CK5003的源码和ISA仍在尾部把越界预取改为stage0读取。现有四级ring+预装一阶段额外5stage，相对必要D2L请求K512为15.6%、K768为10.4%、K2048为3.9%，是请求数比例，不是可保证的耗时收益。','',
 '**等待位置需要重建正确性。** CK5003每级在部分VMEM等待后额外执行lgkmcnt(0)再barrier，这是P6b精度实验要求的。BLAS多个主循环保留lgkmcnt(4/6/8)，与8/16/32条MMAC burst及各自ring映射配套。直接删除CK等待或抄一个waitcnt数值没有正确性依据。K32增加每级工作量/隐藏等待是迁移候选，性能尚未验证。','',
 '**不是靠更少寄存器、无group扫描或硬件BF16打包取胜。** 所选BLAS的VGPR都不少；有标量逐组累加tile数的查找循环；可见软件RNE转换序列。CK小M的非MMAC VALU/wave在历史低CTA诊断中约243.5，BLAS约531/774，不能预设BLAS固定指令一定少。静态kernel字节数含activation/尾块/epilogue备选分支，不等于动态工作量。','',
 '**cache modifier存在可隔离差异。** 所选BLAS D2L没有slc/glc；CK NN A的idxen D2L带slc。已验证的是ISA位差异，尚未证明其在gfx936上的性能作用。不能仅凭这个差异宣称L2失效根因或预报收益。','',
 '## 4. 新鲜profiler证据','',
 '| ID/路径 | 逻辑tile数 | waves | MMOP总数 | HBM读MB | L2命中% | L1→L2平均周期 | DRAM credit stall |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in ps:
  out.append(f"|{r['id']}/{r['backend']}|{r.get('ctas',r.get('logical_tiles'))}|{r['waves']:.0f}|{r['mmop_insts']:.0f}|{r['hbm_read_MB']:.2f}|{r['l2_hit_pct']:.1f}|{r['tcp_tcc_read_latency']:.0f}|{r['dram_credit_stall_frac']:.3f}|")
 out += ['',
 '所有26个dispatch的MMOP均与逻辑padding模型一致。4003是80个常驻CTA/320个硬件wave迭代3920逻辑tile，每CTA平均49个tile；其余V7/BLAS为每tile一个CTA。对4003不能把logical tile当成新的硬件CTA，不能直接用其per-wave指令数和非persistent路径比较。','',
 '### N1232：回退实现的代价，不是补零乘加量','',
 'ID146两边MMOP完全相同（12,042,240），CK/BLAS HBM读量仅相差约9%，而单launch耗时相差约6.42倍。CK总VMEM 22,067,200 vs BLAS 2,007,040（11.0倍），非MMAC VALU 112,251,200 vs 4,997,216（22.5倍），LDS指令11,226,880 vs 4,515,840（2.49倍）。4003的code-object元数据另确认VGPR256、private132B、VGPR spill39/SGPR spill106；BLAS无spill。已验证回退通用路径存在大量额外指令/寄存器溢出，未把6.42倍逐周期分摊给各项。仅将尾块padding压缩不会消除差距。BLAS有行列边界处理与专门写回路径，CK仍按既有契约落4003。建议以后以独立tail实例承接，并保持已验收V7 epilogue；用户此前要求做法确定前不要动，本阶段没有实现。','',
 '### 小M中长K与新增M128/256','',
 'ID772的旧CK反而快于5003，5003耗时+18.31%；MMOP降为旧路径1/4，但HBM读增加4.6%，L1→L2平均延迟702→879周期，DRAM credit stall 0.125→0.238。BLAS用128×128×16、16条MMAC/级，5003用64×128×16、8条/级。已验证减少padding乘加没有改善这个shape的访存效率；未隔离证明额外lgkmcnt(0)是唯一根因。128×128、四wave的新几何是候选，尚未做其CK A/B；不能从BLAS已快直接推出移植必快。','',
 '新增1186/1187的BLAS同为128×128×16，5003分别产生2倍于BLAS的CTA/wave：1186为1024 vs512 waves，1187为2048 vs1024 waves。M256上5003与原CK矩阵指令相同（524,288），waves却为原CK4倍（2048 vs512），HBM读36.27 vs35.13 MB，耗时+19.89%。新增波数、重复panel加载和固定工作是回退解释的实证项，尚未逐项隔离耗时；M128/256继续保留原路径。','',
 '### 大M大K','',
 'ID1178两边tile都是256×256×16、512线程、64KB LDS、8192 CTA/65536 waves、1,073,741,824条MMOP，同卡单launch仍约1.95倍差距。CK HBM读45.741GB vs BLAS9.471GB（4.83倍），L2命中45.95% vs89.80%，L1→L2平均延迟756 vs378周期，TCP alloc stall 0.831 vs0.606；支持缓存复用/数据请求组织是主要诊断方向。工作量、wave数量和LDS冲突比例接近，不能把问题归为小M tile。当前CK已是M01=8，BLAS为WGM8，不能再次只建议“改成8”。下一项优先隔离NN A加载的slc modifier，再检验地址/推进节奏；modifier作用仍是待验证假设。','',
 '## 5. 全1193目标的历史覆盖清单（不是本阶段全量结果）','',
 '下面来自P6c增量/历史拼接表，BLAS旧卡0/新增卡4、CK卡4；5003策略仍是实验模拟，未提升默认。','',
 '| 属性分组（互斥） | 总数 | R<0.99 | R<0.8 |',
 '|---|---:|---:|---:|']
 for r in coverage['cohorts']:out.append(f"|{r['cohort']}|{r['count']}|{r['below_099']}|{r['below_08']}|")
 out += ['',
 '各组总和1193，R<0.99共463条、R<0.8共66条。不能把代表9条的新鲜测试外推为1193均已接近BLAS。以每shape各调用一次且只累加正差额作诊断，prefill约152.8ms、N1232约51.9ms，远大于短K小M的0.24ms；这不是生产调用权重，只说明全数据目标必须保留大M与tail两条路线。','',
 '## 6. 验证与复现','',
 f"- 成功计时{vs['timing_jobs']}项、profile {vs['profile_jobs']}组/26 dispatch、SQTT1次，共89次GPU命令；89次门禁HCU4=0.0%，无重试/失败。所有CK计时CPU抽样/guard通过。第一轮8个Native中小shape全输出CPU参考与guard通过，共{vs['cpu_full_blas_checks']:,}元素，failed=0/max_abs=0；1178沿用已冻结精度验证。没有新kernel，没有重新跑selector/smoke/fused，不把本阶段CK抽样说成全输出。",
 '- SQTT成功采得681/5003第一活跃CU的169,920B token。所谓raw文本实际是16位十六进制token，不是已解码的PC/latency表；现场Linux未使用可验证解码器，提供的XCompute为Windows安装包，因此本阶段未声称逐PC时延归因。探针到此停止，不扩大不可分析的采集；文件可交XCompute打开。SQTT下事件耗时含工具开销，不纳入计时表。',
 '- scripts/manifest、逐条命令/门禁、所有轮次、原始计数和失败（如有）均归档。受保护readfirstlane diff MD5仍4c7f49a5c7592767d53195a713db6b8a，未stage；二进制与shape文件SHA保持。原始BLAS ISA/code object和perf/SQTT只留现场，未复制进CK仓库。','',
 '```bash',
 'ulimit -c 0',
 'python3 /workspace/v7_coverage/p6d/elf_isa.py',
 'python3 /workspace/v7_coverage/p6d/isa_features.py',
 'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6d/run.py timing',
 'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6d/run.py profile',
 'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6d/run.py trace',
 'python3 /workspace/v7_coverage/p6d/analyze.py',
 '```','',
 '关键输出：ISA_FACTS_PASS 11 kernels; vendor_tail_blocks 6；75项计时全部成功；13组26 dispatch MMOP全部匹配；各GPU命令前HCU4=0.0%。归档脚本再次验证所有输入/二进制SHA和保护diff。正式逐条命令以jobs.jsonl.gz为准；重放脚本会跳过已经成功的key，不能把空重放声称为新复测。','',
 '## 7. 下一阶段候选与停止点','',
 '1. 低CTA小M：独立64×128×32（必要时64×64×32）pipeline，重建K32物理panel、两次K16矩阵子步、operand寄存器和等待；与现有5003同shape做精度和A/B，优先681/633/76及新增短K。K%32及N%Tn契约和回退必须显式，默认仍关闭。',
 '2. K尾段：独立验证no-global-load drain，调整剩余VMEM的等待阈值，保留已证明必要的LDS完成等待；不能只加if跳过load却保留原vmcnt常数。把这项和K32几何分别A/B，分清收益来源。',
 '3. 中长K/较多CTA：128×128四wave（含wave排列选择），针对772及新增M128/256验证；K/CTA等属性选核，避免shape白名单，不把小tile全面放行。',
 '4. 大M大K：优先保持相同tile、只对NN A的slc modifier做独立实例/二进制的单因素实验，先1178及一条已接近BLAS的对照；再看地址组织/推进节奏，不重复全量。该项针对已验证的4.83倍HBM读差距，优先级高于盲加M tile变体；不能改动受保护readfirstlane文件。N1232独立实例须先完成用户此前保留的方案选择。','',
 '本阶段为诊断，没有“新优化后全数据分桶”。按/workspace/CLAUDE.md每阶段完成后本地commit并停下汇报；下一阶段应以这里的证据选择一个隔离实验，不能宣称已实现全部shape目标。','']
 (ROOT/'report.md').write_text('\n'.join(out))
 print('REPORT_WRITTEN',ROOT/'report.md')

if __name__=='__main__':main()
