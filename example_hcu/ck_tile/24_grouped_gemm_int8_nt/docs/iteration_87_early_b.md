# Iteration87: early versus late next-B prefetch

Keep256x256x128,uniform eight C waves,A batch4,record0/1024,both mappings and the existing one-barrier schedule. Late control reads next B before current high-half MMAC. Early policy waits current B,reads next B into the free opposite static register slot before low MMAC,then consumes low/high. The final stage never prefetches beyond K. No extra B buffers,changed math,group count,layout or synchronization count. CPU register-slot model44 scenarios PASS;not hardware proof.

240 full-range INT8/FP64 checks and56 representative performance-output/guard checks PASS. All8 kernels retain214VGPR,64KB LDS,no scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median.

All28 same-mapping/same-record pairs regress with early prefetch. The speedup ratio(early throughput divided by late throughput) ranges from0.7170 to0.9116.

| M | N | K | G | best candidate | early B | TOPS | target TOPS | R |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
|16384|2048|512|8|29110|False|179.706|201.009|0.8940|
|4096|1024|2048|64|29100|False|361.784|374.064|0.9672|
|2048|1536|2048|8|29000|False|406.516|414.146|0.9816|
|16384|3072|4096|48|29000|False|540.448|574.971|0.9400|
|16384|4096|6144|8|29010|False|574.376|636.582|0.9023|
|2048|6144|7168|96|29010|False|606.512|575.891|1.0532|
|2048|4096|8192|128|29000|False|626.522|601.440|1.0417|

Own-code disassembly isolates original orientation/record1024. Static instruction counts across the whole function(including fallback/tail) match:320MMAC,12 indexed B loads,16 A-to-LDS loads,64 LDS reads,11 barriers. Native loop order differs. Both versions have the explicit current-B vmcnt(0) before low A reads;an additional generated vmcnt(0) appears immediately before the first low-half MMAC. In late it precedes issuing future B;in early the future B loads now occur before that extra wait,so they are drained before low MMAC. Actual current and future B register ranges are distinct. This defeats the intended overlap in the generated loop body;static inspection does not by itself allocate a runtime percentage to the stall.

Next isolate compiler-visible current-B waiting. Replace only that wait in an independent factorial experiment with __builtin_amdgcn_s_waitcnt(0x0f70),verify it actually disassembles to vmcnt(0),and compare asm/builtin x early/late x record0/1024 x both mappings. Preserve all other barriers/waits and production code. The current CK INT8 MMAC primitive uses __builtin_hcu_mmac_i32_16x16x32_i8;its existing implementation and generated-code evidence are reference points,not grounds to remove waits without validation. Public v80 formal975/1178 remains unchanged,with203 open fixed-target misses.
