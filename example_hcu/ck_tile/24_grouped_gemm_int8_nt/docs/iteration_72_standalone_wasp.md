# Iteration72: standalone WASP kernel and packed fallback

Move the passing one-barrier/A-prefetch pipeline into root-level typed problem/pipeline/epilogue/kernel headers. Replace the experimental geometry fallback dependency with the example's production packed-load/MMAC components. All12 waves participate in loads/barriers; the first8 compute the fallback4x2 MMAC grid. Masked loads cover K16-vector tails; scalar epilogue supports both original/equivalent-transposed views and padded output strides. Native stride/span guards remain in the kernel. No experimental include remains in these five root headers.

114 full-range INT8/FP64 boundary checks and24 representative output checks PASS across six development instance configurations. This includes new K16384/padded-stride fallback and M257/K128 profiles, in addition to previous ragged/tail and odd/even stage profiles. The explicit development probe may exercise K128; the public ABI retains its existing K>=256 contract. Input clones/guards pass in every boundary check. Metadata shows214VGPR for batches2/4 and232 for batch8,64KB LDS,zero scratch/spill. GPU7,3 HIP-event samples,warmup2/repeat5,median. Public registration and default selection follow in a separate change.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|18401|36.627960|540.331|574.971|0.9398|
|16384|4096|6144|8|18401|13.208702|499.449|636.582|0.7846|
|2048|6144|7168|96|18401|33.615612|515.157|575.891|0.8945|
|2048|4096|8192|128|18401|28.078716|626.531|601.440|1.0417|

The standalone wrapper retains representative throughput close to the previous experimental native path. Resource metadata changes because the generic fallback body changed; no claim of identical machine code is made. This is a prerequisite validation for public integration, not all1178 acceptance. Next register three shape-rule instances and run CPU selection replay plus public boundary and performance validation. Full fixed-target optimization remains incomplete.
