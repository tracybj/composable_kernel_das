# P6f cache factor checkpoint — 实验用，不合入

This is a four-shape factor study, not completion of P6f or a1193-shape GPU replay. All experiments are default off. T0/deadlines remain in session.json. Physical GPU4, official selector/benchmark, warmup10/samples30/single launch, no argument build. Native frozen performance reference is GPU0/repeat20; precision reference is separate per-group hipBLASLt.

Added5011/5012 for SLC/NoSLC M01=5; reused M01=1/8 instances. Geometry256×256×16,512threads,64KiB LDS and all waits/epilogue unchanged. Old11 shader bodies are byte-identical; paired cache variants differ only in SLC bits. v1 public ABI is unchanged except appended enum IDs.

36 accuracy jobs /60 whole-output comparisons /31,798,646,784 elements passed with max_abs=0 and guards.120 formal jobs yield20 complete three-round same-epoch contrasts. Selector2880 new checks and14316 existing-selection equivalence cases passed; official query/selector/smoke passed in epoch1.

| ID | control ms | NoSLC M01=5 ms | paired change | frozen Native ms | R after |
|---|---:|---:|---:|---:|---:|
| 781 | 79.783798 | 52.795570 | -33.83% | 63.276226 | 1.199 |
| 782 | 42.939903 | 26.377674 | -38.57% | 28.935217 | 1.097 |
| 783 | 20.830704 | 13.199207 | -36.66% | 14.154627 | 1.072 |
| 784 | 9.724089 | 6.621372 | -31.92% | 7.053823 | 1.065 |

All round medians and CVs are retained in cache_comparisons.json. ID781 control changes58.32→79.78→86.53ms while the candidate stays52.79–52.80ms, so its estimated percentage benefit is within observed variation and is explicitly uncertain. No clean slow result is attributed to MKFD without event evidence. Direct M01=5 candidate medians are2.66–3.85% faster than M01=8 NoSLC; percent improvements against separate controls are insufficient to rank candidates.

Four clean epoch1 profiler captures show identical actual dispatch geometry and MMOP count. ID781 original HBM reads92.14–96.28GB, NoSLC M01=1 51.70GB, NoSLC M01=5 31.49GB; L2 hit48.81–51.27%→78.87%, TCP→TCC read latency665–809→349cycles. These support cache traffic as a material cause; profiler-derived time is not formal timing.

Environment events: the initial regression pre-gate and a later SLC-M5 profiler saw hymgr service transitions. The latter returned rc0 but is invalid (jobs ok=false); raw log retained. Completed formal data ended before that event. Two bounded recoveries passed known original5000 whole-output probes; epoch2 and cumulative recovery618.48s are recorded at this checkpoint. No reset/restart/switch-to-evade was used.

The12-case attribute domain was fixed before expansion: common/nonragged BF16 NN gfx936,1024<M<=2048,N>=4096,N%256=0,3072<=K<=4096,K%16=0,G<=1024 plus resource bounds. NoSLC with M01=5 for N/256>16, otherwise1. Expansion and an entire fresh ID781 three-round contrast are running separately; no adoption claim for that domain yet. Tail and decode implementations remain CPU prototypes and are not validated on GPU in this checkpoint.

Reproduce: python3 /workspace/v7_coverage/p6f/campaign.py cache_accuracy; cache_native; cache_perf; then analyze_cache.py / isa_audit.py / selector_check.py. Frozen plans refuse binary changes. Jobs and exact command/environment/binary SHA, raw logs and script snapshots are archived; vendor ISA/perf/binaries remain outside Git.
