# Iteration99: frozen cached-weight public API acceptance,all1178

PhysicalGPU7(PCI0000:f5:00.0),gfx936,INT8 NT→INT32,no scale. The approved expert-weight preparation is reused across GEMM launches;original FP16/BF16 ABI/defaults are unchanged. Service recovery restored HIP availability before this run. This is a new full measurement through the isolated public API,not a splice of development screens.

46 full-range INT8/FP64 boundary checks PASS,covering routes[0, 1, 3, 4, 5],padding,tails,ragged/empty groups,nondefault streams,packed bytes,input immutability and output guards. Eight shared-weight/explicit same-address refresh/multiple-stream/queued-work-destruction checks PASS. Original public-context nine source hashes remain unchanged. Prior CPU compatibility evidence:general selector80,INT8 selector45,andold symbol-address coexistence checks pass. No new FP16/BF16 GPU regression claim is made here.

All1178 original M/N/K/G and phases are preserved;1178/1178 full-output numerical checks PASS,with bounded128-row exact FP32 integer references for inputs[-8,8],TF32disabled,guards and sampled input immutability.10warmups,50kernel repeats,3samples,median HIP-event GEMM time. No result reuse from v80/v96.

| Scope | Cases | R>=1 | R>=0.99 | R>=0.95 | minimumR | medianR |
|---|---:|---:|---:|---:|---:|---:|
|All|1178|1093|1124|1166|0.866067|1.215079|
|decode|872|853|861|865|0.866067|1.333154|
|prefill|306|240|263|301|0.907754|1.037899|

85 targets remain unmet. Historical v80 reached975/1178 on a separate run;these are separate-run attainment counts,not paired speedup measurements. All-target completion is not achieved. The full current CSV is `current_cached_v97_1178.csv`;`iteration_99_remaining.csv` lists exact gaps. G and E are aliases for the original nonempty expert/group count.

Preparation submit-host time,host time until ready,and the GPU event interval around the full create operation are separate from GEMM time. The event interval includes allocation/metadata/host scheduling gaps and is not pure pack-kernel timing. Extra persistent weight bytes are reported;raw source weights remain valid and immutable for the handle lifetime. Preparation is one observation per shape,not a tuned-cost median.

API library SHA256 `e39601a17a85e8392ae83197c71bdd1af86c2c2510fc71e35ac9231d5a159ac3`. Source revision `18ecabaf7ee73f925398faceb58af6bdf8e6a0d1`. Full raw records,manifest,progress and audited CSV are under `results/cached_api_v97_all_1178/`. Boundary/lifetime evidence is copied alongside this report. Existing raw-NT public v80 remains available;the cached-weight interface is additive.
