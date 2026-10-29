# Iteration89: four cooperative waves with N-fragment reuse

Independent CK-Tile Problem/Pipeline/Kernel/Epilogue.128x256x64 CTA,four waves,each covers128x64 using8 M16 by4 N16 accumulators. All waves load A directly into two swizzled LDS slots;no dedicated producers. B remains the original unmodified NT physical layout with record0/1024 policies. A register batch1/4 and both internal orientations are screened. This isolates task division from DeepGEMM's physical weight packing and gfx938 matrix-load instructions;neither is silently assumed available.

CPU model covers8192 A bytes exactly and33 stage-parity protocols. GPU248 full-range INT8/FP64 checks PASS,including1/3/5/7/9-stage native cases,mixed groups,padded strides,tails and input/output guards. Eight kernels use180 or204VGPR,16KB LDS,zero scratch/spill.84 representative same-build output/guard checks PASS,including four existing WASP controls. GPU7,three HIP-event samples,warmup2/repeat5,median. Screening data is not the public1178-case acceptance CSV.

| M | N | K | G | new TOPS | same-build best WASP TOPS | speedup |
|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|176.791|179.181|0.9867|
|4096|1024|2048|64|326.040|361.453|0.9020|
|2048|1536|2048|8|350.095|407.750|0.8586|
|16384|3072|4096|48|312.017|540.366|0.5774|
|16384|4096|6144|8|209.336|574.296|0.3645|
|2048|6144|7168|96|224.211|606.730|0.3695|
|2048|4096|8192|128|271.889|626.268|0.4341|

No tested shape improves against the same-build WASP controls. Lower resources do not compensate for this implementation's repeated B traffic and synchronization/data-supply path. For an aligned256x256 output over K128,the old geometry requests32KB A+32KB B;four128x256x64 tiles request32KB A+64KB B(50% more requested global operand bytes). These are nominal requests,not measured HBM traffic. Do not claim this result excludes all cooperative or four-wave implementations.

Next isolate DeepGEMM-style physical B packing while retaining the successful256x256x1288C4P compute schedule. Measure conversion separately and report one-use cost;static-weight amortization is an explicit possible interface extension,not free preprocessing for the existing raw-NT benchmark. Public v80 remains unchanged at975/1178 fixed-target passes and203 open misses.
