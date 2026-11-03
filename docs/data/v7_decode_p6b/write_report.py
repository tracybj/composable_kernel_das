#!/usr/bin/env python3
"""Write measured tables after analyze.py and validate.py have passed."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'stage_wait'


def main():
    perf = json.loads((DATA / 'perf_summary.json').read_text())
    guard = json.loads((DATA / 'guard_summary.json').read_text())
    profiles = json.loads((DATA / 'profile_summary.json').read_text())
    validation = json.loads((DATA / 'validation.json').read_text())
    assert len(perf) == 36 and len(guard) == 6 and len(profiles) == 12
    lines = ['# P6b：小 M V7 双 tile 原型实验', '',
        '生产代码基线 8b05927；诊断阶段提交 67e45fa。新实例 5003/5004 仅由实验开关接入默认 selector，默认关闭；本阶段不提升为生产默认。没有运行 1178 全量，没有 push，N=1232 继续走原路径。', '',
        '## 实现与契约', '',
        '- 新流水：M64、N128/64、K-stage16、256线程/4waves、四级32KB LDS ring。每lane累加器32/16个FP32，编译VGPR80/55、SGPR47/51；scratch/spill均为0。编译日志的静态LDS=0，实际launch动态申请32768字节。',
        '- MMAC、Direct-to-LDS、三组operand bank和12-phase环沿用V7，按新tile重建地址与读取映射。物理A128×16、B16×128，逻辑tile以外通过无效INDEX填0。N64 epilogue行XOR掩码限制到3，避免原来的7使4个16B单元越界/重叠；独立epilogue保持BF16 RNE、行谓词和16B写回。',
        '- 只支持gfx936、BF16 NN、common M∈[1,256]、common K∈[64,4096]且K%16=0、G≤1024、无ragged/empty，并要求N%Tn=0、tile数量受原上限约束。不满足条件走旧selector。N=1232不满足整tile契约，保持4003。',
        '- 实验开启：CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1，加 CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M=128 或64；CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M=1优先。未设置、值非法、总V7禁用均不进入新实例。',
        '- 公开v1结构体和GemmTransKernelArg布局不变；新增枚举ID，不改已有ID。热路径没有新增GPU分配、设备回读或host侧设备同步。按架构/算子属性选核，保留回退，不使用shape白名单。', '',
        '变更文件：新增 gemm_pipeline_v7_small_m.hpp、grouped_gemm_v7_small_m_epilogue.hpp、small_m launch helper和两份实例TU；更新selector/开关/接口ID、device_args分派、provider、CMake与独立V7/V8源清单（保留专用编译选项）、selector/query测试。原 gemm_pipeline_v7_uniform.hpp 的用户readfirstlane修改保持未提交，未纳入本阶段变更。', '',
        '## 同步问题及修正证据', '',
        '初版N64在ID76上出现偶发数值错误，冻结二进制三次重跑分别失败1/8/5个CPU抽样点；N64 K64 G4也复现。N64 K768 G1和N128 K64 G8对照通过。仅在最后两步排空LDS仍失败（tail_wait/），因此当时停止性能采集。', '',
        '每阶段在VMEM部分等待之后、block barrier之前加入lgkmcnt(0)，让look-ahead LDS读取先完成，再允许下一步消费operand bank及复用ring。32KB与临时64KB配置分别在K768 G4、K64 G4上连续20次全输出比对全部通过（80次，stage_wait_control/）。最终移除64KB调试开关与冗余尾部等待，保留每阶段完成等待，使用32KB。此结果支持阶段间LDS完成/复用时序不足；尚未逐指令区分bank就绪和跨wave覆盖，不将更细的因果判断写成已验证。', '',
        '## 执行与正确性', '',
        '所有GPU命令物理卡4串行；每条前hy-smi确认HCU%=0，core dump关闭。正式数据均由device_args_benchmark和默认selector产生，没有强制实例ID。正确性用backend=both与指定hipBLASLt库全输出比较；性能warmup=10、samples=30，每样本单次launch。', '',
        f"`CAMPAIGN_PASS valid_jobs={validation['valid_jobs']}`：smoke6、accuracy48、regression5、perf216、guard36、profile12。54个精度用例各3次全输出比较，共{validation['full_output_comparisons']}次、{validation['checked_elements']}个元素，failed=0、max_abs=0。覆盖M=1/7/32/63/64/65/127/128/129/255/256，K64/80/768/4096，G32/1024，超额tile slot，以及N1232/prefill/large-K/ragged/empty/禁用回退。", '',
        '`SELECTOR_TABLE_PASS cases=80 registry=23`；`SELECTOR_V7_BOUNDS_PASS cases=11`；`SELECTOR_V7_M01_8_PASS cases=8`；`SELECTOR_V7_SMALL_M_PASS cases=50`；`DEVICE_ARGS_QUERY_PASS registry=23`；`DEVICE_ARGS_SMOKE passed=21 cases=21 negative=1`；fused NN/TN CPU参考检查均PASS（原实例）。fused的容差检查不等于上述全输出逐位一致检查。', '',
        '## 三轮交替性能', '',
        '18个代表shape，每轮每条base→N128、base→N64，合计216次。修复前/后都在物理卡4，使用三轮各自median的中位数；变化=(after/before−1)。下表BLAS是卡0冻结基线且采用另一launch口径；R=BLAS卡0/新CK卡4，仅供历史对照，不能作为同卡加速比。完整轮间值、CV与min/max保留在JSON和原日志。', '',
        '| ID | M,N,K,G | Tn | 原CK卡4 ms | 新CK卡4 ms | 变化 | BLAS卡0 ms | R |',
        '|---|---|---|---|---|---|---|---|']
    for r in perf:
        lines.append(f"| {r['id']} | {r['M']},{r['N']},{r['K']},{r['G']} | {r['tile_n']} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['delta_pct']:+.1f}% | {r['blas_card0_ms']:.6f} | {r['R']:.3f} |")
    lines += ['', '代表集合汇总（不代表872条decode或149条尾部的整体覆盖）：']
    for tile in [128, 64]:
        rows = [r for r in perf if r['tile_n'] == tile]
        lines.append(f"- Tn={tile}：18条变化中位数{statistics.median(r['delta_pct'] for r in rows):+.1f}%，慢于原版>2%的{sum(r['delta_pct']>2 for r in rows)}条。")
    lines += ['', '## 默认路径对照', '',
        '实验开关关闭，新旧二进制三轮交替。原5000、5001、5002的三个shader及三个descriptor全部SAME（kernel_compare_final.txt），原readfirstlane差异MD5仍为4c7f49a5c7592767d53195a713db6b8a。机器码一致不能代替下表计时；若存在>2%的行，按实测保留。', '',
        '| ID | 原CK卡4 ms | 新二进制默认卡4 ms | 变化 | 三轮before ms | 三轮after ms |',
        '|---|---|---|---|---|---|']
    for r in guard:
        b = '/'.join(f'{v:.6f}' for v in r['before_round_ms'])
        a = '/'.join(f'{v:.6f}' for v in r['after_round_ms'])
        lines.append(f"| {r['id']} | {r['before_ms']:.6f} | {r['after_ms']:.6f} | {r['delta_pct']:+.2f}% | {b} | {a} |")
    lines += ['', '## Profiler结构核验', '',
        'ID76/681/633/358，各原版/N128/N64一次GC采集，每次两次dispatch。24/24个dispatch的MMOP和wave数均匹配补齐tile模型；一轮派生指标仅辅助解释，不当作性能计时。固定项为VALU−MMOP每wave的指令数，不能解释为耗时占比。', '',
        '| ID | Tn（0=原版） | CTA | waves | MMOP指令 | CU busy | HBM读MB | 固定VALU/wave |',
        '|---|---|---|---|---|---|---|---|']
    for r in profiles:
        lines.append(f"| {r['id']} | {r['tile_n']} | {r['ctas']} | {r['waves']:.0f} | {r['mmop_insts']:.0f} | {r['busy_cu_frac']:.3f} | {r['hbm_read_MB']:.2f} | {r.get('valu_minus_mmop_insts_per_wave', 0):.2f} |")
    lines += ['', '## 重放与证据', '',
        '工作目录 /workspace/v7_coverage/p6b；正式campaign为stage_wait/，失败初版与tail_wait/分开保留，不能混入成功汇总。bench_base是修改前冻结二进制；最终benchmark SHA256=72e701492a3419ce552aeb0376e2c6c95151a375080a76939507fc8d1a38f548，只改注释的最后一次重编前后相同。原始日志/.perf留在工作目录，不放入Git。', '',
        '```bash',
        'ulimit -c 0',
        'cmake --build build --target tile_example_grouped_gemm_device_args_benchmark tile_example_grouped_gemm_selector_test tile_example_grouped_gemm_device_args_query_test tile_example_grouped_gemm_device_args_smoke tile_example_grouped_gemm_fused_test -j6',
        '# 每个mode逐次运行；脚本内逐条检查卡4空闲并串行执行',
        'HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6b/run_experiment.py smoke --campaign stage_wait',
        '# 同样依次运行 accuracy、regression、perf、guard、profile',
        'python3 /workspace/v7_coverage/p6b/analyze.py',
        'python3 /workspace/v7_coverage/p6b/validate.py',
        'python3 /workspace/v7_coverage/p6b/write_report.py',
        'python3 /workspace/v7_coverage/full1178_p4b/retest/kcmp.py /workspace/v7_coverage/p6b/bench_base build/bin/tile_example_grouped_gemm_device_args_benchmark V7Uniform',
        '```', '',
        '所有单条命令/选核/结果见stage_wait/jobs.jsonl，门禁见gate.jsonl；perf_summary.json、guard_summary.json保存三轮原值；profile_summary.json和raw_counters.json保存所有派生指标及原始计数。build2/build5/build6_final/build7_manifest.log保留构建证据。', '',
        '本阶段保留四级流水，尚未实现短K浅流水或参数调度优化。320CTA是历史经验阈值，104/45覆盖率不能直接解释为本次收益覆盖。N=1232尾块支持、完整尾部验收及默认选核提升均留待用户确认。', '']
    lines[4:4] = (ROOT / 'conclusions.md').read_text().splitlines() + ['']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    print('REPORT_WRITTEN', ROOT / 'report.md')


if __name__ == '__main__':
    main()
