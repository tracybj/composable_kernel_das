# Iteration86: single-publication C/CP consumer schedule

Keep the public256x256x128 geometry,physical NT operands,exact INT32 output,record0/1024 policies,A batch4 and four producer/eight consumer waves. In an independent experimental namespace,testC/CP counts0/8,4/4,8/0 with both mappings. The production uniform-C single-barrier assertion remains unchanged.

The new CP runner consumes previous high-half MMAC from captured registers in I1,then reads current low A. I2 computes current low MMAC,reads/waits current high A and prefetches next B. One I2 barrier publishes next A and releases current LDS;the high MMAC uses registers after release. The final high half drains from B register slot0 for odd stages,slot1 for even stages. Thus current A LDS reads finish before release,previous high registers are consumed before overwrite,and all roles have one publication/reuse barrier per stage. CPU stage-generation/contribution/barrier model covers66 scenarios;it is not hardware proof.

360 full-range INT8/FP64 checks PASS,including native1/3/5/7/9-stage cases,padding,tails,ragged groups and fallback.84 representative performance outputs/guards PASS. GPU7,three HIP-event samples,warmup2/repeat5,median. Source and library identities frozen.

| M | N | K | G | best candidate | C waves | ms | TOPS | target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|28108|8|1.525952|180.135|201.009|0.8962|
|4096|1024|2048|64|28108|8|3.026943|363.242|374.064|0.9711|
|2048|1536|2048|8|28008|8|0.252512|408.215|414.146|0.9857|
|16384|3072|4096|48|28008|8|36.642708|540.113|574.971|0.9394|
|16384|4096|6144|8|28018|8|11.491677|574.074|636.582|0.9018|
|2048|6144|7168|96|28018|8|28.537561|606.825|575.891|1.0537|
|2048|4096|8192|128|28008|8|28.084442|626.403|601.440|1.0415|

All8 configurations that include CP report420--488B private/scratch and nonzero VGPR spill fields,with256 declared VGPR. The4 all-C controls retain214VGPR and zero scratch/spill. Register-carry CP variants are much slower,so the tested schedule is not promoted despite passing these numerical checks. Register/latency tradeoffs remain dependent on compiler and implementation;this is not a proof that every CP protocol must be slow.

Next test an earlier B prefetch within the existing all-C runner:after waiting for current B,issue next B into the free opposite register slot before low-half MMAC,then consume low and high. The late control issues next B only before high. No extra buffers or barriers,no high-A carry,and no changed geometry/work. This targets B latency coverage with a smaller lifetime change. Public v80 formal975/1178 remains authoritative and203 misses remain open.
