# Iteration74: cache-swizzle stride eligibility ablation

Retain the public one-barrier/A-prefetch pipeline and generic packed fallback. Compare mode0 (original power-of-two stride/64 eligibility),mode1 (enable descriptor bit30 for every nonzero64-byte-multiple stride),mode2 (disable that bit),with A batch4 in both internal mappings. A producer descriptors still pass allow_swizzle=false; only the B structured-buffer descriptor can change. No pointer,row stride,group count,arithmetic or timing-contract change. These are development instances; the public selector stays unchanged.

132 full-range INT8/FP64 boundary checks and24 representative performance-output checks PASS. Additional contiguous K3072/6144/7168 profiles exercise the broadened mode,along with padded inputs and generic fallback. All boundary inputs and guards are preserved. All six kernels use214VGPR,64KB LDS,no scratch/spill. GPU7,3 HIP-event samples,warmup2/repeat5,median. Passing these samples is not proof that arbitrary cache-mode settings are universally safe.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|20000|36.614296|540.532|574.971|0.9401|
|16384|4096|6144|8|20000|13.209406|499.422|636.582|0.7845|
|2048|6144|7168|96|20000|33.591354|515.529|575.891|0.8952|
|2048|4096|8192|128|20000|28.084955|626.392|601.440|1.0415|

Mode1 does not improve any representative best over the original mode0. Disabling cache-swizzle materially regresses the power-of-two-stride K4096/K8192 cases,while K6144/K7168 show no meaningful gain when broadened. The original policy remains the fastest representative choice. Preserve this negative result and do not promote mode1 or lower fixed targets. No new public throughput records are substituted for current_public_v73_1178.csv. All1178-target optimization remains incomplete;263 actual-default gaps are documented separately.
