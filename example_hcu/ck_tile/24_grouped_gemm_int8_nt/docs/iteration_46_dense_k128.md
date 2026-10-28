# Iteration46: dense K128 copy tile and VGPR prefetch

gfx936 physical GPU7; INT8 NT -> INT32,no scale. Use contiguous K128 global16-byte copies, one64KiB dense LDS operand tile, packed VGPR prefetch of the next tile and typed hcu_buffer_load_asm. Compare16-byte XOR/b128 paired-MMAC reads with8-byte XOR/b64 reads, all-wave copies vs4 producers, axis grouping, masking and slice fences. Mathematically the paired b128 fragments cover each K64 chunk with two exact native MMAC operations.

72 full-range INT8/FP64 combinations and44 performance-output records pass, covering aligned/tail/ragged/padded-stride cases and full inputs/guards in the boundary tests. Timing:3 HIP-event samples,warmup2/repeat5,median. This remains a development experiment and does not establish the entire supported-domain contract or all1178 throughput.

Resource evidence: the all-wave paired variant declares244 VGPRs and no spills. The4-producer paired variant declares256 VGPRs,32 private bytes and7 VGPR spills; the masked version has6 spills. Unpaired4-producer variant has253 VGPRs and no spills. Thus at least part of the paired producer regression has direct spill evidence. Next separate producer/consumer loops and accumulator/prefetch lifetimes.

Known proof gap before promotion: the masked native offset guard covers actual K but not all rounded K128 padding offsets near the signed-i32 boundary. The normal-stride measurements here do not exercise that edge. Keep this historical experiment unpromoted; the split-role kernel in iteration48 includes padded K in its stride-span proof.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|8402|44.722485|387.217|
|2048|6144|7168|96|8602|45.401402|381.427|
|2048|6144|7168|96|8901|41.451256|417.775|
|2048|6144|7168|96|8902|64.635094|267.924|
|2048|6144|7168|96|8903|41.324986|419.052|
|2048|6144|7168|96|8904|49.141884|352.394|
|2048|6144|7168|96|8905|58.147736|297.816|
|2048|6144|7168|96|8906|45.446747|381.046|
|2048|6144|7168|96|8907|41.717049|415.113|
|2048|6144|7168|96|8908|69.695183|248.472|
|2048|6144|7168|96|8909|41.434681|417.942|
|2048|4096|8192|128|8402|42.901306|410.062|
|2048|4096|8192|128|8602|42.688187|412.109|
|2048|4096|8192|128|8901|43.027672|408.857|
|2048|4096|8192|128|8902|64.041237|274.701|
|2048|4096|8192|128|8903|43.114807|408.031|
|2048|4096|8192|128|8904|50.602966|347.651|
|2048|4096|8192|128|8905|59.162865|297.352|
|2048|4096|8192|128|8906|45.630039|385.540|
|2048|4096|8192|128|8907|43.861557|401.084|
|2048|4096|8192|128|8908|71.930290|244.573|
|2048|4096|8192|128|8909|43.010998|409.016|
|16384|3072|4096|48|8402|53.065685|372.957|
|16384|3072|4096|48|8602|56.084881|352.880|
|16384|3072|4096|48|8901|51.907410|381.279|
|16384|3072|4096|48|8902|70.346481|281.339|
|16384|3072|4096|48|8903|52.684853|375.653|
|16384|3072|4096|48|8904|63.429237|312.020|
|16384|3072|4096|48|8905|72.237358|273.975|
|16384|3072|4096|48|8906|58.479633|338.429|
|16384|3072|4096|48|8907|52.872536|374.319|
|16384|3072|4096|48|8908|72.310478|273.698|
|16384|3072|4096|48|8909|52.055187|380.197|
|16384|4096|6144|8|8402|18.077852|364.926|
|16384|4096|6144|8|8602|17.953724|367.449|
|16384|4096|6144|8|8901|17.036125|387.240|
|16384|4096|6144|8|8902|24.429466|270.046|
|16384|4096|6144|8|8903|17.430042|378.488|
|16384|4096|6144|8|8904|20.187740|326.786|
|16384|4096|6144|8|8905|23.867001|276.410|
|16384|4096|6144|8|8906|17.976091|366.991|
|16384|4096|6144|8|8907|17.303707|381.252|
|16384|4096|6144|8|8908|25.842875|255.276|
|16384|4096|6144|8|8909|17.031355|387.348|

All-wave dense K128 improves some representatives to roughly381-419 TOPS but does not close the BLAS gap. Mixed producer/consumer loops regress, particularly when spilling. No default change.

Library SHA256:d4a0093f92a2a24056b081b9809df71216669302d133c3c6dec4fa53f0060006
