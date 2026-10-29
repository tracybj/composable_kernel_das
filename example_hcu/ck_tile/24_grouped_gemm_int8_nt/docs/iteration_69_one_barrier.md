# Iteration69: one stage-publication/reuse barrier

Starting from the passing A-prefetch pipeline, compare the original two-barrier stage schedule with one end-stage barrier. Uniform-C consumers finish H0/H1 A reads and MMAC before the single barrier; producer waves finish the following A stage copy and vmcnt wait before that same barrier. At release, the following A slot is ready and the old slot has no consumers. Initial publication is producer fill0/wait/barrier; each remaining stage is producer fill-next/wait/barrier; final producer drain matches the last consumer barrier. Register-prefetched A fragments are all consumed before stage release. OneBarrier requires ConsumerSplit8 at compile time, so the older CP persona cannot accidentally use this schedule. Masked generic fallback remains unchanged.

Compare batches2/4/8 and original/equivalent-transposed internal mapping, A prefetch enabled in every candidate.12 instances include identical same-build two-barrier controls.204 full-range INT8/FP64 boundary checks (odd/even stage counts, padded/contiguous inputs, ragged/tails through fallback) and48 representative performance-output checks PASS. All12 kernels use243-244VGPR,64KB LDS,no scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|18401|36.486427|542.427|574.971|0.9434|
|16384|4096|6144|8|18401|13.203453|499.647|636.582|0.7849|
|2048|6144|7168|96|18401|33.603355|515.345|575.891|0.8949|
|2048|4096|8192|128|18401|27.992346|628.464|601.440|1.0449|

The single-barrier native path improves every representative best relative to the same-build two-barrier controls. Original internal mapping, batch4,candidate18401 is fastest on all four tested representatives. ID1177 reaches628.46TOPS versus frozen601.44TOPS, but the other three remain below their individual targets. This is a four-shape screen, not evidence that all344 large-M or all1178 workload targets are reached. No public default promotion.

Own-kernel static ISA summaries show16 versus10 s_barrier instructions for matched full kernels, with MMAC/LDS-read/wait counts unchanged. The counts include fallback and epilogue; do not interpret their ratio as a dynamic time percentage. Next screen all344 original M>=256 shapes with six single-barrier candidates and the current-default-instance control, then derive dimension-based rules and validate the actual public selector.
