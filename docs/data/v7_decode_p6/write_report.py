#!/usr/bin/env python3
"""Validate the completed four-shape campaign and render its review artifact."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    jobs = [json.loads(s) for s in (ROOT/'jobs.jsonl').read_text().splitlines()]
    good = {r['key']:r for r in jobs if r['ok']}
    assert len(good) == 56, len(good)
    summary = json.loads((ROOT/'summary.json').read_text())
    derived = json.loads((ROOT/'derived.json').read_text())
    assert len(derived) == 56
    for r in summary:
        assert r['timing']['rounds'] == 3 and r['gc']['rounds'] == 3 and r['tlb']['rounds'] == 1
        assert r['gc']['all_mmop_count_matches_padded']
    for r in derived:
        if r['suite'] != 'timing':
            assert r['dispatch_count'] == 2, r
        elif r['backend'] == 'blas':
            assert r['failed'] == 0 and r['max_abs'] == 0 and r['guards_ok']
        else:
            assert r['instance_id'] == 5000
        if r['suite'] == 'gc' and r['backend'] == 'ck' and r['id'] in [76,681,633]:
            assert r['valu_minus_mmop_insts_per_wave'] == 709.75, r
            assert r['vmem_above_two_loads_per_kstage_per_wave'] == 10.25, r
    by = {(r['id'],r['backend']):r for r in summary}
    lines = [
        '# P6a：decode 四条 shape 的 profiler 诊断', '',
        '日期：2026-11-03（Asia/Shanghai）。生产代码基线：v7-coverage，8b05927；保留未提交 readfirstlane，预期 diff md5：4c7f49a5c7592767d53195a713db6b8a。', '',
        '本阶段完成 §4.1 的诊断；没有改 kernel、selector、epilogue 或实例，没有重编，没有运行 1178 全量，也没有 push。N=1232 留待用户决定。', '',
        '## 执行与复核', '',
        '所有 GPU 子命令固定物理卡 4，HIP_VISIBLE_DEVICES=4、CUDA_VISIBLE_DEVICES=4，core dump 关闭；每条之前用 hy-smi 确认 HCU%=0，串行运行。run.py 沿用 p4a 的 GC/TLB section 和 p4b/gate4.sh 的三次重试规则，逐条写 jobs.jsonl，支持断点续跑。', '',
        '正式计时：四条 shape，CK→BLAS 交替三轮；两边 warmup=10、samples=30，每个 sample 一次 launch。CK 走合规 benchmark 和默认 selector，全部选中 5000；BLAS 用现有 Native 桥接库固定重放冻结基线算法，核对 algorithm_id、kernel_name、workspace_bytes 和 BLAS SHA256，不重新调优。', '',
        '计数器：三轮 GC 对照（24 条）和一轮 TLB/UMC 对照（8 条），每条 warmup=1、sample=1，解析最后一次匹配 dispatch；两次 dispatch 均留存于 raw_counters.json。加上 24 条正式计时，共 56 条有效记录。', '',
        '```bash', 'ulimit -c 0',
        'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6/run.py timing',
        'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6/run.py profile',
        'python3 /workspace/v7_coverage/p6/analyze.py',
        'python3 /workspace/v7_coverage/p6/write_report.py', '```', '',
        '证据：manifest.json（文件哈希）、jobs.jsonl（完整命令和结果）、gate.jsonl（门禁输出）、runs/（原始日志/.perf）、raw_counters.json、derived.json、summary.json。先前 P4a ID966 的解析复核：GUI=8236719、MMOP=268435456、waves=32768、CU busy=0.979461、L1→L2=304.594，与历史表吻合。', '',
        '## 计时结果', '',
        '表中卡 4 数值为三轮各自 median 的中位数；括号是三轮 median 的范围。冻结 BLAS 在卡 0，采用 20 次连续 launch，单列展示，不能和新的单次 launch 口径混用。采集中的 xprof PERF/RESULT 包含重放开销，不作为性能数据。', '',
        '| ID | M,N,K,G | CK 卡4 ms（范围） | BLAS 卡4 ms（范围） | 同卡 R | 冻结 BLAS 卡0 ms |',
        '|---|---|---|---|---|---|'
    ]
    for case_id in [76,681,633,358]:
        c,b=by[case_id,'ck'],by[case_id,'blas']
        def timing(r):
            t=r['timing']; lo,hi=t['median_ms_range']
            return f"{t['median_ms']:.6f}（{lo:.6f}–{hi:.6f}）"
        lines.append(f"| {case_id} | {','.join(map(str,c['shape']))} | {timing(c)} | {timing(b)} | {b['timing']['median_ms']/c['timing']['median_ms']:.3f} | {b['frozen_blas_card0_ms']:.6f} |")
    lines += ['', '小 kernel 的 CV 和轮间差异较大。完整的 30 样本值（BLAS）、CV/min/max（两边）保留在 jobs.jsonl，不删除离群值，不使用最快一轮替代三轮中位数。这组计时用于诊断，不构成新实例的回退验收。', '',
        '## 并行度与实际执行量（已验证）', '',
        'BLAS 桥接在 groupgemm_bridge.cpp 的 Context::problem() 中设 m=N、n=M，因此 MT 的前两维需交换为逻辑 M/N。所有 GC dispatch 的 MMOP 数均与补齐后的 tile 计算量匹配，不能拿有效输出 2MNKG 直接核验 decode 的 MMOP。', '',
        '| ID | CK/BLAS 逻辑 tile | CK/BLAS CTA | CK/BLAS waves | CK/BLAS MMOP 指令 | CK/BLAS HBM 读 MB |',
        '|---|---|---|---|---|---|']
    for case_id in [76,681,633,358]:
        c,b=by[case_id,'ck']['gc'],by[case_id,'blas']['gc']
        tile=next(r['tile'] for r in derived if r['suite']=='gc' and r['id']==case_id and r['backend']=='blas')
        lines.append(f"| {case_id} | 256×256×16 / {'×'.join(map(str,tile))} | {int(c['ctas'])}/{int(b['ctas'])} | {int(c['waves'])}/{int(b['waves'])} | {int(c['mmop_insts'])}/{int(b['mmop_insts'])} | {c['hbm_read_MB']:.2f}/{b['hbm_read_MB']:.2f} |")
    lines += ['',
        '1. 三条问题 shape 的 CK 只有 24/32 CTA，80 个 CU 上同时活跃 CU 的数量上限是 30%/40%。这是由实际执行量核验支持的结构性并行度限制。',
        '2. 同三条 shape，CK 的 MMOP 数都是 BLAS 的 4 倍，实际 HBM 读量却接近。M tile 对实际计算量有独立影响；应评估减少补零行运算对每个 CTA 成本的改善。某个 V7 小 tile 的最终收益仍需实测。',
        '3. ID358 的两边都具备充分 CTA，读约 0.81 GB，性能接近。它与三条小 shape 的差异符合并行度不足的判断；四点仍不足以对两种因素做独立的时间归因。', '',
        '## 指标汇总（三轮中位数）', '',
        '| ID/后端 | CU busy | 平均 wave/CU | MMOP 活跃相对值 | L1→L2 周期 | DRAM credit stall | wave 平均生命周期（周期） |',
        '|---|---|---|---|---|---|---|']
    for r in summary:
        c=r['gc']
        lines.append(f"| {r['id']}/{r['backend']} | {c['busy_cu_frac']:.3f} | {c['mean_resident_waves_per_cu']:.3f} | {c['mmop_active_per_simd_cycle']:.4f} | {c['tcp_tcc_read_latency']:.1f} | {c['dram_credit_stall_frac']:.4f} | {c['mean_wave_lifetime_cycles']:.1f} |")
    ratios=', '.join(f"ID{i} {by[i,'ck']['gc']['mean_wave_lifetime_cycles']/by[i,'blas']['gc']['mean_wave_lifetime_cycles']:.2f}×" for i in [76,681,633,358])
    lines += ['', 'CK/BLAS 平均 wave 生命周期比：'+ratios+'。三条问题 shape 的 wave 自身也更慢，支持将单 CTA 成本与并行度同时评估；该比值不是两项损失的可加时间分解。']
    lines += ['',
        '公式沿用 P4a：CU busy=SQ_BUSY_CU_CYCLES/(GRBM_GUI_ACTIVE×80)；MMOP 活跃=SQ_ACTIVE_INST_MMOP/(GUI×80×4)，只作相对值；L1→L2=TCP_TCC_READ_REQ_LATENCY_sum/TCP_TCC_READ_REQ_sum；DRAM credit stall=TCC_EA_RDREQ_DRAM_CREDIT_STALL_sum/(GUI×16)。', '',
        '计数器来自多个 replay pass；小 kernel 的派生比值受 pass 间波动影响。GUI/1.6GHz 与 AQL 时长在小 shape 上不一致，例如 ID76 第1轮 CK 为 65.872 µs / 38.080 µs，BLAS 为 28.021 µs / 20.800 µs。因此不以 GUI 换算时间做精确的损失分解，完整范围和 AQL 时间保存在 derived.json/summary.json。稳定的 CTA、wave、指令计数是主要结构证据。', '',
        '## 短 K 的固定项', '',
        'CK 的 ID76（K768）、ID681/633（K2048）均有 (SQ_INSTS_VALU−SQ_INSTS_MMOP)/wave=709.75；对应 VALU 计数占比 31.6% / 14.8%。这是固定指令项占比随 K 缩短而上升的证据，不能读成耗时占比。', '',
        '源码 kNumStages=4、kPrefetchDist=5：prologue 先加载 4 个 stage，再预装第 5 个；k-loop 每步仍发 A/B 两次加载，末端越界预取改读 stage 0。计数器相应为 VMEM/wave=2×(K/16)+10.25（ID76/681/633，包含输出写回）。相对必需的 K/16 个 stage，额外 5 个 stage 加载在 K768 时是 10.4%，K2048 时是 3.9%；重复地址可命中缓存，不能等同于 HBM 流量增加或可回收的耗时。', '',
        '**已验证**：固定指令/预取项在短 K 上占比更大；三条问题 shape 同时存在低 CTA 和更大的实际计算量。**仍是假设**：把 4 级改浅能带来多少收益，以及固定项中 pipeline、参数/地址设置、epilogue 各占多少时间。本次没有做改流水的 A/B 实验，不将其写成已确认的修法。', '',
        '## 对后续方案的约束与建议', '',
        '并行评估 N 切分与短 K 固定开销；同时把 M tile 对实际 MMOP/LDS/寄存器工作量的影响纳入，小 tile 需按 V7 重建和实测，旧 4002 更慢的结论保持。', '',
        '用户给出的 104/45 条统计按“是否达到 320 CTA”纳入评估。但 320 是经验分段，不是硬件的成功/失败界线：三条问题 shape 仅切 N 到 64，CTA 可从 32/24/32 增到 128/96/128，仍小于 320，却已经跨过 80 CU 的数量。因此 45 条未达阈值不等于 N 切分必然无收益；104 条达阈值也不是性能收益保证。两条路线均需独立实测。', '',
        '未来实例若无列谓词，必须将 N % Tn == 0 写入支持契约、selector 和启动前拒绝/回退条件。保持 M 尾块、ragged/empty 语义与接口，不按 shape 白名单选核。N=1232 本阶段未处理。', '',
        '三条问题 shape 的 CK DRAM credit stall 均为 0，全部 CK 的 UTCL2 miss 均为 0；对照 ID358 的 DRAM credit stall 约 0.13。结合读量接近理论输入量，当前证据支持优先处理并行度和每 CTA 成本，未观察到 P4a 大 K 那种 HBM 读放大/拥塞模式。', '',
        '本阶段到此停止，待用户确认后才进入方案实验。', '',
        '## 变更与验证范围', '',
        '新增的只有诊断脚本、报告和小型证据文件。未改任何生产源码、CMake/provider 或实例编号。BLAS 正式计时 12/12 次全输出参考校验 failed=0、max_abs=0、guards_ok=true；CK 正式计时 12/12 次 CPU 抽样 failed=0、guard=0。没有因生产改动而需要执行的新实例正确性/性能验收；本阶段没有重跑 selector/device_args/fused 回归。', '',
        '保留两条准备阶段失败记录：第一次 CK 输出字段识别错误（实际 benchmark PASS），第一次 BLAS 库加载冲突（native bridge 尚未运行）。修复诊断脚本后续跑，二者不计入有效数据，没有删除原始日志。', '',
        '复核结果：有效作业=56，三轮 GC 的补齐 MMOP 模型全部吻合，全部 profile 文件有两次 dispatch；阶段完整性检查通过。',
    ]
    tlb = ', '.join(f"ID{i}={int(by[i,'ck']['tlb']['utcl2_miss'])}/{int(by[i,'blas']['tlb']['utcl2_miss'])}"
                    for i in [76,681,633,358])
    lines += ['', '补充 TLB 证据（CK/BLAS 的 UTCL2 miss）：'+tlb+'。UTCL2 hit 和 UMC page hit 原值见 summary.json。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print('CAMPAIGN_CHECK_PASS valid_jobs=56 timing=24 gc=24 tlb=8 padded_mmop=24/24')


if __name__ == '__main__':
    main()
