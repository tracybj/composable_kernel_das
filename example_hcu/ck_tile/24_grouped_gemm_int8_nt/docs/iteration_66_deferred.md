# Iteration64: complete short-live INT8 WASP kernel

Defer only the final AReadBatch fragments of each nonfinal H1 half to the next stage. All A LDS reads complete before the stage release barrier; the copied A fragments and previous B H1 registers remain live until the following C_I1, which consumes them before waiting for the next B stage. The final stage computes its H1 immediately, so no deferred work remains at exit. Compare1/2/4/8/16 fragments and original/equivalent-transposed internal mappings with CTA group4. External NT inputs and actual G are unchanged; generic masked fallback remains.

170 full-range INT8/FP64 boundary checks and40 representative performance-output checks PASS. Batches1-8 use243-244VGPR with no scratch/spill. Batch16 uses256VGPR,164B private segment and76 spilled VGPRs in each orientation. Physical GPU7,three HIP-event samples,warmup2/repeat5,median.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|13081|44.005398|449.745|574.971|0.7822|
|16384|4096|6144|8|13081|15.630013|422.077|636.582|0.6630|
|2048|6144|7168|96|13081|40.096794|431.888|575.891|0.7499|
|2048|4096|8192|128|13081|34.769081|505.972|601.440|0.8413|

The measured deferred variants do not improve representative best performance over iteration65. Batch16 drops to roughly108-142TOPS on the second representative while the nondeferred family reaches510TOPS there. Spilling correlates with the severe regression; this is not a cycle attribution proof. Smaller deferred batches remain correct but add no stable measured gain in this screen. Do not promote this variant, lower frozen targets or substitute these oracle screens for full1178 acceptance. Next evaluate priority/scheduling in the no-spill nondeferred family, whose consume_short path currently omits the mmac_half priority scope of the original WASP source.
