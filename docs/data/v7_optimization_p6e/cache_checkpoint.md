# P6e checkpoint: isolated NN A cache hint (experiment only)

Eight-hour campaign starts 2026-11-04 06:53 UTC, experiment cutoff12:53, deadline14:53. This checkpoint records completed cache isolation; expanded validation and small-M experiments remain ongoing. Experimental switches default off. No full1193 replay or N1232 changes.

## Causal isolation

New5005/5006 keep the old256x256x16 geometry, M01=1/8 respectively, epilogue, wait sequence and address mapping. Only A direct-to-LDS SLC is removed. gfx936 ISA comparison confirms51 static sites per shader differ only in the SLC machine bit, identical28400/28580-byte shaders and VGPR212/SGPR47. Original five shaders are byte-for-byte unchanged. The user's uncommitted readfirstlane diff remains unstaged (MD54c7f49a5c7592767d53195a713db6b8a).

Enable global CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1 and CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC=1. The disable switch wins; old small-M takes precedence. Default selector requires gfx936/BF16/NN, common nonragged/nonempty M/K, whole N256 tiles, K>=64%16, G<=1024 and tile bounds. Candidate applicability is broader than demonstrated benefit; the predeclared expansion policy is K>4096.

## Completed measurements

Physical card4 only, every job gated with independent hy-smi and host MemAvailable>=64GiB, serial common lock. New code first passed13 full-output CK-vs-per-group-BLAS accuracy jobs. Selector/query/smoke pass. Initial ten-shape timing:3 alternating base/new rounds, warmup10 samples30, single launch, include-args-build0, CPU accuracy/guards each job. No MKFD encountered. One retained harness failure: query inherited global V7 enable; corrected harness environment and reran query. This was not a kernel failure.

|ID|Original CK ms|No-SLC ms|change|
|---|---:|---:|---:|
|76|0.035840|0.035680|-0.45%|
|358|0.662562|0.660642|-0.29%|
|962|2.786242|2.786402|+0.01%|
|966|5.484966|5.492649|+0.14%|
|1095|0.429680|0.410401|-4.49%|
|1118|9.911280|9.709521|-2.04%|
|1126|19.669907|19.193344|-2.42%|
|1164|22.523037|21.811846|-3.16%|
|1177|115.906080|59.503242|-48.66%|
|1178|56.744381|29.821995|-47.45%|

Fresh paired profiler ID1178: HBM40.306→11.595GB, L2 hit52.92→87.75%, TCP-to-TCC read latency695.6→371.7cycles. ID1126: HBM12.890→8.415GB, L2 hit73.93→83.60%; ID962:1.875→1.553GB, timing flat. MMOP/waves/geometries match all12 dispatches. Counters are diagnostic, not a precise decomposition of time.

ID1178 frozen Native BLAS card4 single-launch P6d29.029898ms vs new29.821995ms (R≈0.973); near parity is not >=0.99. Old card0 repeat20 baselines remain separate. No claim that all1193 shapes meet the long-term goal.

## Reproduction and evidence

From workspace, `ulimit -c 0`, then HIP_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 python3 /workspace/v7_coverage/p6e/run.py <regression|cache_accuracy|cache_perf|cache_profile> (each sequentially). Harness resumes completed keys. Analyze with analyze_cache.py and cache_isa.py. Full commands, hash-pinned runner versions and failure logs: /workspace/v7_coverage/p6e/{jobs.jsonl,gate.jsonl,runs,scripts}. Original benchmark72e70149…; measured cache benchmark1052b95c681371b0713c357ade4e3ede659df08dd4d763e3cf6023e9ab5d0d2d.

Remaining: all212 K>4096 shapes plus new15 and controls; M01 walk cross-check; small-M K32 and no-global-load drain independent precision/performance validation. No production defaults promoted. Local commit only, experiment must not be merged as accepted production change.
