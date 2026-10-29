# Iteration68: overlapping A LDS reads with MMAC

Compare identical A batches2/4/8 and original/equivalent-transposed internal mapping, with and without A-register ping-pong. The prefetched path loads the first batch, waits for its LDS completion, issues the following batch into the other register buffer, computes the current batch, then exchanges register roles. Every prefetched batch is waited before consumption; the final batch issues no next read. Stage barriers, B prefetch timing and external INT8 NT / INT32 no-scale contract remain unchanged.12 candidates, no case-ID dispatch.

204 full-range INT8/FP64 boundary checks and48 representative performance-output checks PASS. All12 kernels use243-244VGPR,64KB LDS,no scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median. All boundary guard/full-input clone checks pass. Same-build controls share the batch and orientation.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|17401|39.641560|499.254|574.971|0.8683|
|16384|4096|6144|8|17411|14.464029|456.102|636.582|0.7165|
|2048|6144|7168|96|17211|37.193787|465.597|575.891|0.8085|
|2048|4096|8192|128|17411|30.743610|572.223|601.440|0.9514|

A prefetch improves the representative best throughputs to roughly466,572,499,456TOPS for IDs1158,1177,902,1038 respectively. Gains are measured without profiler; the own-kernel ISA summaries show unchanged static operation counts with changed register/scheduling placement. Static counts alone do not prove overlap cycles. Fixed targets remain unmet and no public default changes are made. Next test whether the uniform-C pipeline can safely replace its midpoint and end-stage barriers with one end-stage publication/reuse barrier, using a same-build two-barrier control. Full1178 target acceptance remains incomplete.
