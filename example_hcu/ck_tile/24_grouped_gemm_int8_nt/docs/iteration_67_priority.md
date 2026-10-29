# Iteration67: MMAC priority scope comparison

Compare priority mode0 (unchanged),mode1 (priority3 after the first MMAC in each A batch, reset0 after that batch),mode2 (priority3 after the first MMAC in a half-stage, reset0 at half-stage completion). Each mode shares the same A batch2/8/16, original/equivalent-transposed internal mapping, CTA group4 and external INT8 NT / INT32 no-scale contract.18 candidates, no case-ID dispatch.

306 full-range INT8/FP64 boundary checks and72 representative performance-output checks PASS. All18 kernels use243-244VGPR,64KB LDS,no scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median. The unchanged control in the same build provides the priority comparison, not a different-version oracle. Boundary guard and full input clones pass.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|15600|43.393078|456.091|574.971|0.7932|
|16384|4096|6144|8|14811|15.053053|438.255|636.582|0.6884|
|2048|6144|7168|96|14811|38.635098|448.227|575.891|0.7783|
|2048|4096|8192|128|15610|34.468437|510.385|601.440|0.8486|

Per-batch priority yields roughly2% gains on two representatives, while the other two are fastest with mode0. No candidate meets the fixed target; these measurements do not establish a universal priority setting. Whole-kernel ISA summaries verify that MMAC/LDS/wait counts stay equal within a given batch/orientation while priority instructions change. Static counts include fallback and epilogue, not dynamic cycle percentages. No default promotion; all1178 acceptance remains incomplete. Next isolate A-read/MMAC overlap using two small register buffers and identical no-prefetch controls.
