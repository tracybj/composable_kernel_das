#!/usr/bin/env python3
"""Render only from persisted measurements, never from a per-shape winner."""
import collections,datetime,json,statistics,subprocess
from pathlib import Path
import run as r
P=r.ROOT

def load(name):return json.loads((P/(name+'.json')).read_text())
def main():
 audit=load('final_audit');assert audit.get('summary',{}).get('passed'),'Completion audit must pass before final report'
 summary=load('summary');cache=load('cache_expanded');small=load('small_representatives');new=load('new15');native=load('native_reference');profiles=load('small_profiles');resources=load('small_resources');walk=load('cache_walk');session=load('session');decisions=load('experiment_decisions')
 jobs=[json.loads(x) for x in (P/'jobs.jsonl').read_text().splitlines()];good=[x for x in jobs if x.get('ok')];gates=[json.loads(x) for x in (P/'gate.jsonl').read_text().splitlines()];waits=[json.loads(x) for x in (P/'waits.jsonl').read_text().splitlines()] if (P/'waits.jsonl').exists() else []
 recovery={x['environment_epoch']:x for x in (json.loads(line) for line in (P/'recovery.jsonl').read_text().splitlines()) if x.get('ok')}
 outage_seconds=sum((datetime.datetime.fromisoformat(recovery[event['environment_epoch']+1]['at'])-datetime.datetime.fromisoformat(event['began_at'])).total_seconds() for event in load('environment_transitions')['events'])
 primary=[x for x in cache if x['policy_applies']];c1178=next(x for x in cache if x['id']==1178);n1178=next(x for x in native if x['id']==1178)
 now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(session['start_utc']);elapsed=(now-start).total_seconds();cs=summary['cache_expanded'];a=audit['summary'];current=load('small_binary')
 lines=['# P6e：大 M 缓存复用与小 M 流水实验（实验用，不合入）','',f'开始 {session["start_utc"]}；报告生成 {now.isoformat()}；已用壁钟 {elapsed/3600:.2f} 小时。新实验截止12:53 UTC，硬截止14:53 UTC。原始数据在 `/workspace/v7_coverage/p6e/`。本轮授权自主跨阶段，仅本地提交，不 push。','',
 f'缓存隔离取得明确收益：1178耗时下降{abs(c1178["delta_pct"]):.2f}%，212条长K范围中位变化{cs["median_delta_pct"]:+.2f}%，无>2%回退。低CTA的K32在21条完整属性范围相对既有5003再改善{abs(summary["small_expand_k32_lowcta"]["median_delta_pct"]):.2f}%中位数，无>2%回退；高CV结果仍按不确定处理。','',
 f'drain并非普遍有效：189条中位变化{summary["small_expand_drain_shortk"]["median_delta_pct"]:+.2f}%，总耗时变化{summary["small_expand_drain_shortk"]["sum_delta_pct"]:+.2f}%，{summary["small_expand_drain_shortk"]["regress_gt2"]}条回退>2%、最差+{summary["small_expand_drain_shortk"]["max_regress_pct"]:.2f}%。square扩大100条后中位变化{summary["small_expand_square_m128_256"]["median_delta_pct"]:+.2f}%，{summary["small_expand_square_m128_256"]["regress_gt2"]}条回退>2%、最差+{summary["small_expand_square_m128_256"]["max_regress_pct"]:.2f}%，本版不采用为通用策略。实验默认关闭，新增15条已纳入，长期1193全部接近BLAS的目标尚未完成。','',
 '## 测量边界','',
 'CK 前后均为物理卡4、合规 device_args_benchmark、默认 selector、include-args-build=0、warmup10、samples30，每个样本单次 launch。每条比较保留三轮相邻原版→新版；时间列为三轮测量中位数，变化为三个同环境阶段内配对耗时比的中位数，未跨恢复前后配对。二者在波动时不一定等于时间列直接相除；不取最快轮、不逐 shape 选最优核。计时前通过精度；每次性能作业另有 CPU 抽样及 guard。','',
 '冻结 BLAS 旧1178条来自卡0的 Native GroupedGemm repeat20；新增15条冻结基线来自卡4、同样repeat20。本轮另在卡4重放8条精确冻结算法，分别报告repeat1和repeat20。历史R=冻结repeat20/CK单launch，含跨卡/跨协议，不能当成同口径直接胜负。','',
 f'本轮记录 {a["attempts"]} 次原始尝试、{a["successful"]} 个接受项目（另保留环境流程不完整而排除的原始测量），测量了 {audit["no_1193_full_replay"]["detail"]["unique_ids"]} 个不同正式ID；没有运行1193全量。全输出 CK-vs-逐组BLAS比较 {a["full_comparisons"]} 次，共 {a["full_elements"]:,} 元素，max_abs集合={a["max_abs_errors"]}。这不代表每条扩大范围shape都做过全输出比较，扩大计时范围是CPU抽样+guard。','',
 '## 大 M：A 的 SLC 提示确实损害这批长 K 的缓存复用','',
 '5005/5006仅去掉NN A Direct-to-LDS的SLC，分别保留M01=1/8。原256×256×16、环形LDS、等待、epilogue和地址组织不变。ISA逐shader确认51个静态加载位置仅SLC机器位不同，原五个5000…5004 shader字节保持一致。VGPR212/SGPR47不变。','',
 f'主策略预先固定K>4096，扩大212条全部做了三轮A/B；另测新增15条和4个短K控制，共231条。212条中位耗时变化 {cs["median_delta_pct"]:+.2f}%，三轮各调用一次总耗时的中位数 {cs["sum_before_ms"]:.3f}→{cs["sum_after_ms"]:.3f}ms，逐轮配对总耗时变化中位数 {cs["sum_delta_pct"]:+.2f}%，改善>2% {cs["improve_gt2"]}条、回退>2% {cs["regress_gt2"]}条。R≥0.99为 {cs["R_ge099_before"]}→{cs["R_ge099_after"]}。这些统计只对应212条。','',
 '|ID|原CK ms|去SLC ms|变化|冻结BLAS ms/卡|历史R新版|','|---|---:|---:|---:|---:|---:|']
 for i in [1178,1177,1126,1164,1118,1095,962,966]:
  x=next(x for x in cache if x['id']==i);lines.append(f'|{i}|{x["before_ms"]:.6f}|{x["after_ms"]:.6f}|{x["delta_pct"]:+.2f}%|{x["blas_ms"]:.6f}/卡{x["blas_gpu"]}|{x["R_after"]:.3f}|')
 lines+=['']
 for phase in ['decode','prefill']:
  x=summary['cache_'+phase];lines.append(f'缓存范围中{phase}有{x["count"]}条，变化中位数{x["median_delta_pct"]:+.2f}%，各调用一次总耗时变化{x["sum_delta_pct"]:+.2f}%，改善>2% {x["improve_gt2"]}条、回退>2% {x["regress_gt2"]}条。')
 lines+=['','初始新鲜配对profiler：1178 HBM40.306→11.595GB、L2命中52.92→87.75%、TCP→TCC读延迟695.6→371.7周期；1126 HBM12.890→8.415GB、L2命中73.93→83.60%。962流量降低但耗时基本持平。12个dispatch的MMOP/waves/几何均匹配。计数器不用于精确拆分损失时间。','',
 f'本轮新鲜同卡4/单launch，1178去SLC {n1178["noslc_ms"]:.6f}ms，冻结Native {n1178["blas_repeat1_ms"]:.6f}ms，R={n1178["R_samecard_single_launch"]:.3f}；repeat20 Native {n1178["blas_repeat20_ms"]:.6f}ms另列。不得把接近BLAS写成全部达标。','',
 '去SLC后再把遍历改回M01=1，在1178/1126/1164相对保留M01=8仍分别变化 '+', '.join(f'{next(x for x in walk if x["id"]==i)["delta_pct"]:+.2f}%' for i in [1178,1126,1164])+'；此尝试不推荐。A索引折叠只完成CPU地址恒等式推演，未在GPU验证，因SLC已证实而未集成。','',
 '## 小 M：四个独立实验','',
 '|开关值|实例|逻辑tile|动态LDS|说明|','|---|---|---|---|---|',
 '|1|5007|64×128×16|32KB|停止无效全局预取；保留operand读取、LGKM0，尾段按未完成加载数采用VMEM2→0|',
 '|2|5008|64×128×32|48KB|重建A64×32/B32×128、K互补子集、DS读取和等待；仍保留旧式冗余尾加载以隔离因素|',
 '|3|5009|128×128×16|32KB|四wave，16-MMAC burst，保留完整LDS完成等待|',
 '|4|5010|64×64×32|32KB|K32的N64几何，保持32KB环以允许两CTA/CU|','',
 'K32不是只改模板数字：N128每stage3个D2L/6个DS；N64为2个D2L/4个DS。A b128切成两组互补K16 operand，B panel对应同一K排列；三bank/12-phase/四stage地址、读写和等待同步重建。所有变体保留此前精度要求的LGKM0，没有直接删除LDS等待。24个初期正确性映射探针采用保守VMEM0，不作为正式性能或末次drain等待版本的验证证据；最终VMEM4→2→0及四个实例以161项正式精度和8项压力、共643次全输出比较为准，全部通过。','',
 '|变体|代表shape数|变化中位数（对其基准）|改善>2%|回退>2%|','|---|---:|---:|---:|---:|']
 for v in ['drain','k32','square','k32n64']:
  x=summary['small_'+v];lines.append(f'|{v}|{x["count"]}|{x["median_delta_pct"]:+.2f}%|{x["improve_gt2"]}|{x["regress_gt2"]}|')
 lines+=['','新shader资源（静态metadata的LDS为0，以下为launch动态LDS；profiler dispatch另行核验）：','',
 '|变体|VGPR|SGPR|动态LDS字节|private字节/线程|','|---|---:|---:|---:|---:|']
 for x in resources:
  md=x['metadata'];lines.append(f'|{x["variant"]}|{md[".vgpr_count"]}|{md[".sgpr_count"]}|{x["declared_dynamic_lds_bytes"]}|{md[".private_segment_fixed_size"]}|')
 lines+=['','17组profiler/34 dispatch的MMOP、waves、真实grid和LDS大小均已核验，scratch为0。以下计数只作结构诊断，不等同耗时占比：','',
 '|ID/变体|VMEM/wave|LDS/wave|MMOP总数|HBM读MB|L2命中%|','|---|---:|---:|---:|---:|---:|']
 for x in profiles:
  lines.append(f'|{x["id"]}/{x["variant"]}|{x.get("sq_insts_vmem_per_wave",0):.2f}|{x.get("sq_insts_lds_per_wave",0):.2f}|{x["mmop_insts"]:.0f}|{x["hbm_read_MB"]:.2f}|{x["l2_hit_pct"]:.2f}|')
 diagnostic=load('square_diagnostic_profiles')
 lines+=['','扩大验证发现square回退后，另做两组原5000→square计数器对照，共4个profile/8 dispatch；其MMOP、waves、grid、LDS及scratch同样核验。下面仅为计数器，不用profile耗时替换三轮正式计时：','',
 '|ID/变体|CTA|waves|MMOP|VMEM总指令|HBM读MB|L2命中%|','|---|---:|---:|---:|---:|---:|---:|']
 for x in diagnostic:
  counters=x['raw_dispatches'][-1];lines.append(f'|{x["id"]}/{x["variant"]}|{x["ctas"]}|{counters["SQ_WAVES"]:.0f}|{counters["SQ_INSTS_MMOP"]:.0f}|{counters["SQ_INSTS_VMEM"]:.0f}|{x["hbm_read_MB"]:.2f}|{x["l2_hit_pct"]:.2f}|')
 lines+=['','934的MMOP减半、VMEM总指令不变，但HBM2226.77→2670.98MB（约+20%），正式计时回退49.63%；714的MMOP不变、waves翻倍、VMEM约1.96倍、HBM约1.99倍，正式计时回退128.16%。支持额外访存和调度成本是这版几何的重要问题，单独减矩阵工作量不足以保证加速。未单独隔离square的A缓存提示或地址组织，不能把回退唯一归因于某一等待指令。','']
 lines+=['','drain/K32N128/square与既有5003比较；K32N64与既有5004比较，用来隔离K32影响。下面同时给出原5000。既有5003列固定取drain的基准臂，各变体变化使用各自紧邻的基准臂；完整数值在逐轮JSON中。','',
 '|ID|原5000 ms|既有5003 ms|drain ms|K32N128 ms|square ms|K32N64 ms|BLAS单launch卡4 ms|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for i in [76,633,681,358,772,1186,1187]:
  row={x['variant']:x for x in small if x['id']==i};b=next(x for x in native if x['id']==i);lines.append('|'+str(i)+'|'+f'{row["drain"]["original_ms"]:.6f}|{row["drain"]["before_ms"]:.6f}|'+ '|'.join(f'{row[v]["after_ms"]:.6f}' for v in ['drain','k32','square','k32n64'])+f'|{b["blas_repeat1_ms"]:.6f}|')
 lines+=['','同卡4的冻结Native算法复测，不重新调优；单launch与每样本连续20次分别计时：','',
 '|ID|Native repeat1 ms|Native repeat20 ms|','|---|---:|---:|']
 for x in native:lines.append(f'|{x["id"]}|{x["blas_repeat1_ms"]:.6f}|{x["blas_repeat20_ms"]:.6f}|')
 lines+=['','适用策略、扩大验证及不采用理由见 `experiment_decisions.json`。这是按M/N/K/G和CTA等属性定义的策略，不是shape ID白名单：','']
 for k,v in decisions.items():lines.append(f'- {k}: '+(v if isinstance(v,str) else v['disposition']+'。'+('范围：'+v['scope']+'。' if 'scope' in v else '')+'后续：'+v['next']))
 lines+=['','扩大范围在首轮扩大测量前固定，三组均测完整属性范围，未按每条shape挑选最快结果：','',
 '|方案|条数|实验基准|配对变化中位数|改善>2%|回退>2%|各调用一次总耗时变化|','|---|---:|---|---:|---:|---:|---:|']
 for name in ['k32_lowcta','drain_shortk','square_m128_256']:
  plan=load('small_expand_plan_'+name);x=summary['small_expand_'+name]
  lines.append(f'|{name}|{x["count"]}|{plan["baseline"]}|{x["median_delta_pct"]:+.2f}%|{x["improve_gt2"]}|{x["regress_gt2"]}|{x["sum_delta_pct"]:+.2f}%|')
 labels=decisions['k32']['metrics']['observational_labels'];lines+=['',f'低CTA K32的21条中，保守描述标签为stable_improvement {labels.get("stable_improvement",0)}条、uncertain {labels.get("uncertain",0)}条；没有将高CV的短kernel称为确定收益。']
 for m in [128,256]:
  x=decisions['square']['by_m'][str(m)];lines.append(f'square的M{m}分层共{x["count"]}条，中位变化{x["median_delta_pct"]:+.2f}%，改善>2% {x["improve_gt2"]}条、回退>2% {x["regress_gt2"]}条。')
 lines+=['','K32低CTA组为M≤64、K128…4096且整除32、N整除128、G≤1024、ceil(M/64)×ceil(N/128)×G<80，共21条。drain为M≤64、K64…<2048且整除16、N整除128、G≤1024，共189条；这两组同时测原5000、既有5003、新变体三个臂，每轮均保留。square为M128…256、K64…4096且整除16、N整除128、G≤1024，共100条，直接与原5000相邻比较。接口正确性范围比这些性能验证范围更宽，不能将实验开关理解成已推荐所有支持shape。','',
 'drain的76对照中，VMEM/wave106.5→96.5，分支103.5→258.5，SALU812.5→974；支持动态尾段判断的控制开销抵消请求减少这一解释，尚未通过独立稳态/尾段分离实验量化。K32/N128将每wave LDS指令和循环控制数降低，但48KB使LDS允许的CTA/CU从2变1；低CTA有收益、高CTA回退与此相符，尚未单独隔离各资源因素。','',
 '完整逐shape、全部轮次、CV、全部回退在 `small_representatives.{csv,json}` / `small_expanded.{csv,json}`；新15条在 `new15.{csv,json}`，含9 decode和6 prefill，prefill开关开启仍选5000。未给未测变体填造时间。','',
 '新增15条G64/N256/K512如下，全部是实测毫秒；六条prefill的实验列为不适用，另列实际开关开启后的5000回退时间。此表的冻结BLAS都是卡4/repeat20，仍与CK单launch不同口径。','',
 '|ID/M|phase|原5000|既有N128|drain|K32/N128|square|K32/N64|prefill回退5000|冻结BLAS|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for x in new:
  values=[x.get(key) for key in ['original_ms','old128_ms','drain_ms','k32_ms','square_ms','k32n64_ms','prefill_switch_on_ms','blas_ms']]
  lines.append('|'+str(x['id'])+'/'+str(x['M'])+'|'+x['phase']+'|'+'|'.join('—' if value is None else f'{value:.6f}' for value in values)+'|')
 lines+=['',
 '小于2%的变化或方向不一致/噪声较大的结果标记uncertain。observational_label采用保守描述规则：三轮相邻变化均超2%且同方向、总体幅度大于最大样本CV的两倍；它不是统计置信区间。所有回退继续计入，原始计时不删。','',
 '## 实现与边界','',
 '新增6个实例5005…5010，实验开关全部默认关闭，接入默认selector/directdispatch，更新CMake专用codegen源清单和provider/query/selector。没有ABI字段变化；原ID和GemmTransKernelArg定义保持。没有热路径新增GPU分配、回读或设备同步。原未提交readfirstlane原样保留，未暂存。','',
 '实验入口：全局 `CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1`；缓存用 `CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=1`；小M用 `CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT=1|2|3|4`，对应上述四个变体。关闭开关优先；小M还受 `CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M=1` 总开关约束。未设置实验开关时沿用已有选核。开启某实验只代表使用其正确性契约，不保证整个支持范围提速；性能建议范围见decisions和扩测表。','',
 '主要源码：`gemm_pipeline_v7_cache_noslc.hpp`隔离缓存提示；`gemm_pipeline_v7_small_m_drain.hpp`实现K16尾段；`gemm_pipeline_v7_small_m_k32.hpp`实现K32两种N宽度；`gemm_pipeline_v7_square128.hpp`实现四wave128平方。六个独立device_args实例、两个launch头、selector和directdispatch负责接入；公开头仅追加实例枚举，CMake/provider源清单同步。代码本地提交1224cca（缓存）及1125bc6（小M正确性）；最终数据单独本地提交，均标记实验用、不合入。','',
 '小M需要gfx936/BF16/NN、common非ragged/nonempty M≤256、G≤1024、K≤4096；drain/square K≥64且K%16=0，K32 K≥128且K%32=0；N必须整除Tn=128或64，epilogue无列谓词。主机tile bound采用真实Tm/Tn。新实例另限制可表示的连续A/B字节范围，所有1193原输入均在范围内。N1232仍4003，未修改其实现或原V7尾块契约。','',
 '新接口守卫不改变已测集合的选核：CPU对1193×12种已有开关组合的实例ID/reason/name共14316条与缓存checkpoint逐字节一致；这不是GPU全量测试。最终ISA再次核验原五shader和两个缓存shader；公开v1头除新增枚举值外其余内容完全一致。','',
 '## 复现与产物','',
 '以下为阶段入口。当前目录按断点跳过已完成项；各阶段应使用对应manifest中的冻结二进制。每条原始命令和环境另见runs中的command.json。','',
 '```bash\nulimit -c 0\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py cache_accuracy\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py cache_perf\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py cache_profile\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py cache_expand --repair-epoch rec2\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py cache_walk\n# 以下每项按序执行；先精度后性能\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/small_run.py regression\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/small_run.py accuracy\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/small_run.py stress\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/small_run.py perf\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/small_run.py profile\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run_native.py --repair-epoch rec3\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/expand_small.py --repair-epoch rec3\nHIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/diagnose_square.py --repair-epoch rec3\npython3 /workspace/v7_coverage/p6e/analyze_all.py\npython3 /workspace/v7_coverage/p6e/scope_analysis.py\npython3 /workspace/v7_coverage/p6e/decide.py\npython3 /workspace/v7_coverage/p6e/observe_environment.py\npython3 /workspace/v7_coverage/p6e/final_audit.py\npython3 /workspace/v7_coverage/p6e/write_report.py\npython3 /workspace/v7_coverage/p6e/archive.py\n```','',
 '脚本按接受的logical_key断点续跑。异常阶段旧值不删除，journalview依据预先记录的invalidations排除并引用新key替代，绝不按快慢选择。环境修复后使用 `--repair-epoch rec1` 续跑缺失及受影响完整A/B；普通新采集可传 `--tag 新标签`。扩大Small需指定已预先固定的 `--plan`。具体命令/环境、二进制及脚本SHA和每次原始输出在 `jobs.jsonl`、`gate.jsonl`、`runs/`、`scripts/`，原BLAS ISA和机器码不入Git。','',
 f'缓存benchmark SHA256=1052b95c681371b0713c357ade4e3ede659df08dd4d763e3cf6023e9ab5d0d2d；最终Small benchmark={current["benchmark_sha256"]}。构建日志build_small.log/build_small_final.log。源码/数据/基线manifest以及final_audit.json构成核验入口。','',
 '## MKFD、失败与限制','',
 f'每条作业前独立hy-smi、仅卡4串行、host MemAvailable≥64GiB；共{len(gates)}条门禁记录、非首次检查{sum(x.get("attempt",1)>1 for x in gates)}条（含恢复观察）。显式等待{sum(x["seconds"] for x in waits):.1f}秒；三段异常起点至恢复探针通过共{outage_seconds/60:.2f}分钟，包含诊断及被排除运行。这些时间全部计入原8小时壁钟，未重置预算。无常驻watch，无服务重启、驱动重载、GPU reset或终止其他会话。','',
 f'本轮在10:15:58 UTC观察到一次hy-smi MKFD（rc=-13）。旧缓存runner仅60秒重试后就继续，缺少完整恢复流程；发现后停止了本任务自己的两个调度脚本，允许在途benchmark自然结束，未终止GPU进程。首轮28个受影响原始项目全部保留并按流程排除，使用新key重新测量完整A/B。已完成两次间隔60秒的健康检查和有三次全输出通过记录的ID76原版探针；恢复后记录environment_epoch。门禁已统一为异常时停止，必须完整恢复后续跑。环境记录与invalidations.json、recovery.jsonl一起归档；该事件不归为kernel精度失败。','',
 '之后ID1140原版在SELECT前报invalid device ordinal，没有精度或性能数据；同期hymgr于10:49:06注销并反复注册到10:49:57。依据这些证据归为疑似环境启动失败，保留失败日志、重做恢复流程并重测跨越服务窗口的完整A/B。全部排除键及证据见invalidations.json/environment_transitions.json/failure_classifications.json，不能把它们当作kernel性能回退。','',
 '第三次事件在11:18:33 UTC：hy-smi退出码0且卡4行存在，但文本含MKFD；新门禁正确停止，未启动下一条小M精度作业。同期11:18:50 hymgr注销。此前57项精度均在异常前完成，无新增性能排除。独立恢复检查及原版探针通过后于11:26:23进入epoch3，重新启动后续测试进程。','',
 'health.py对任何环境转变都停止测试，独立恢复进程须通过相隔60秒的两次检查及已知正确探针，再以新key重测受影响完整A/B；这些观察不保证服务稳定。运行错误不自动归因于环境。','',
 f'最终只读内核日志复核通过：{load("final_environment_observation")["observed_at"]}，{a["successful"]}条接受作业及相应A/B区间均未跨越hymgr注销事件。具体证据在final_environment_observation.json和kernel_environment_final.log。','',
 '本轮失败日志全部保留。初始query一次因脚本泄漏V7 enable与生产默认预期冲突，修正环境后通过；CPU初版ABI文本检查因新增枚举后的逗号/空行过严失败，精确检查修正后通过。新边界测试曾有变量遮蔽导致-Wshadow构建失败，修正后selector/query/smoke通过，失败构建日志保留。实际运行失败明细如下：','']
 for x in summary['jobs']['failures']:lines.append('- '+json.dumps(x,ensure_ascii=False))
 lines+=['','本轮不把局部或历史拼接结果说成1193全部达标。N1232的16条尾部、未测范围、可组合但未验证的K32+drain以及任何不确定微收益仍是后续工作；未改变默认生产选核。每个方向的后续优先级以experiment_decisions.json和实际计数器为准。','']
 (P/'report.md').write_text('\n'.join(lines));print('REPORT_WRITTEN',P/'report.md')
if __name__=='__main__':main()
