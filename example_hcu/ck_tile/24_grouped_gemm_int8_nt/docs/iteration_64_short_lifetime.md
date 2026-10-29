# Iteration64: complete short-live INT8 WASP kernel

Use ordered native I8 MMAC plus one live16-byte A fragment per M microtile in the uniform-C consumer. Keep current WASP double-buffer A direct-to-LDS, structured B loads, quarter-stage producer/consumer barriers and physical external NT layout. Four candidates cover original/equivalent-transposed internal mapping and M CTA groups4/8. Retain generic masked fallback for unsupported/tail cases. Selection IDs are development instance IDs, not workload case dispatch.

68 full-range INT8/FP64 boundary checks (including four repeated smoke profiles) and16 representative performance-output checks PASS. Four kernels use243-244 VGPR,64KB LDS,zero scratch/spills. Physical GPU7,three HIP-event samples,warmup2/repeat5,median. Boundary input clones and guards are checked; representative output verification uses benchmark inputs and guards.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|11004|50.633816|390.869|574.971|0.6798|
|16384|4096|6144|8|11004|18.292126|360.651|636.582|0.5665|
|2048|6144|7168|96|11004|47.214298|366.781|575.891|0.6369|
|2048|4096|8192|128|11004|41.162457|427.384|601.440|0.7106|

This gives a correct full operator configuration but no target-reaching performance gain. More per-fragment LDS waits are a candidate cost; this has not yet been counter-profiled. Ordered MMAC alone failed in62; shortening A lifetime additionally passes in63/64. This does not prove a compiler bug or establish a universal spill/root-cause diagnosis. No public default promotion and no claim of all1178 completion. Next compare small batches of A fragments to trade register lifetime against wait frequency, while preserving the passing dataflow.
