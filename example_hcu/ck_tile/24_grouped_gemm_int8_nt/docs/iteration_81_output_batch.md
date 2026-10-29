# Iteration81: A-register batch and epilogue pairing after record-stride integration

The faster B record1024 path may shift the balance between A prefetch issue and live registers. Separately,the earlier PMC comparison counted more LDS/VALU operations for CK; direct scalar stores remove the output bpermute network but increase store instructions and alter memory transactions. Test these hypotheses together with compatible controls,not by assuming fewer shuffles or more register prefetch must be faster.

Keep the current production WASP pipeline,producer roles,double-buffer LDS,MMAC math,barriers and stride fallback. Sixteen development configurations:original orientation,batches4/8/16,record0/1024,vector versus direct scalar output;plus internal-transposed batches4/8 with record0/1024 controls. Scalar output writes the original MMAC row/column coordinates directly without a wave transpose. All original independent groups included.

352 full-range INT8/FP64 boundary checks and112 representative performance-output checks PASS. GPU7,three HIP-event samples,warmup2/repeat5,median. Resource extraction finds16 kernels,214--233VGPR,64KB LDS,no scratch/spill. Existing public v80 formal results are unchanged.

| M | N | K | G | best development candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|23508|1.521056|180.715|201.009|0.8990|
|4096|1024|2048|64|23504|3.050175|360.475|374.064|0.9637|
|2048|1536|2048|8|23008|0.252544|408.164|414.146|0.9856|
|16384|3072|4096|48|23008|36.624855|540.376|574.971|0.9398|
|16384|4096|6144|8|23108|11.472605|575.028|636.582|0.9033|
|2048|6144|7168|96|23108|28.545979|606.646|575.891|1.0534|
|2048|4096|8192|128|23008|28.094585|626.177|601.440|1.0411|

The existing batch4/vector policies remain best on the four major large-M representatives. Direct scalar output is slower on every tested original-orientation representative;larger A batches do not close the gap despite no spills. Thus do not promote either change. Orientation still matters:the M2048,N1536,K2048,G8 representative reaches408.2TOPS in the original orientation versus329.4TOPS public v80,while M4096,N1024,K2048,G64 benefits from internal transpose. These observations motivate a later dimension-rule review,but a per-shape oracle is not a public rule.

The tested register/output adjustments are insufficient for the remaining gap. Next structural candidate:512x256x64 output/stage geometry,which keeps the A LDS stage at32KB and double-buffer at64KB. Per-stage MMAC work is unchanged(512*256*64 equals256*256*128),but each CTA can reuse B across twice as many M rows. At equal full work,the ideal B tile-load volume halves while A tile-load volume stays equal;consumer accumulators double from128 to256 INT32 registers before other live state,CTA count halves and K-loop count doubles. These costs may erase the reuse gain. This is a hypothesis requiring a new compatible producer/read/fragment/kernel/epilogue implementation and resource/math/performance checks,not a hardware-limit claim or current optimization result.
