# Iteration103: frozen cached-weight public API acceptance,all1178

PhysicalGPU7(PCI0000:f5:00.0),gfx936,INT8 NT→INT32,no scale. The approved expert-weight preparation is reused across GEMM launches;original FP16/BF16 ABI/defaults are unchanged. Service recovery restored HIP availability before this run. This is a new full measurement through the isolated public API,not a splice of development screens.

47 full-range INT8/FP64 boundary checks PASS,covering routes[0, 1, 3, 4, 5, 6, 7],padding,tails,ragged/empty groups,nondefault streams,packed bytes,input immutability and output guards. Sixteen shared-weight/explicit same-address refresh/multiple-stream/queued-work-destruction checks PASS. Original public-context nine source hashes remain unchanged. Prior CPU compatibility evidence:general selector80,INT8 selector45,andold symbol-address coexistence checks pass. No new FP16/BF16 GPU regression claim is made here.

All1178 original M/N/K/G and phases are preserved;1178/1178 full-output numerical checks PASS,with bounded128-row exact FP32 integer references for inputs[-8,8],TF32disabled,guards and sampled input immutability.10warmups,50kernel repeats,3samples,median HIP-event GEMM time. No result reuse from v80/v96.

| Scope | Cases | R>=1 | R>=0.99 | R>=0.95 | minimumR | medianR |
|---|---:|---:|---:|---:|---:|---:|
|All|1178|1117|1141|1176|0.944792|1.252105|
|decode|872|869|869|870|0.944792|1.389190|
|prefill|306|248|272|306|0.958652|1.045743|

61 targets remain unmet. Historical v80 reached975/1178 on a separate run;these are separate-run attainment counts,not paired speedup measurements. All-target completion is not achieved. The full current CSV is `current_cached_v103_1178.csv`;`iteration_103_remaining.csv` lists exact gaps. G and E are aliases for the original nonempty expert/group count.

Preparation submit-host time,host time until ready,and the GPU event interval around the full create operation are separate from GEMM time. The event interval includes allocation/metadata/host scheduling gaps and is not pure pack-kernel timing. Extra persistent weight bytes are reported;raw source weights remain valid and immutable for the handle lifetime. Preparation is one observation per shape,not a tuned-cost median.

API library SHA256 `985b3e3f3f8b337581862457d870dbbabf9ece704482cac82400559b88eb5f54`. Source revision `ce34dec8398acd8250901ee9afcea52f4184f608`. Full raw records,manifest,progress and audited CSV are under `results/cached_api_v103_all_1178/`. Boundary/lifetime evidence is copied alongside this report. Existing raw-NT public v80 remains available;the cached-weight interface is additive.

Previous cached-v97 formal run reached1093/1178 targets,minimumR0.866067. This new independent run reaches1117/1178,minimumR=0.944792. These are separate-run attainment counts,not a paired speedup.
