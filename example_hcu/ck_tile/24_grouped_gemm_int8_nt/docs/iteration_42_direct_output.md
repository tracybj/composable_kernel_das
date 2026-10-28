# Iteration42: output without LDS staging

gfx936 physical GPU7, INT8 NT -> INT32, no scale. Keep aligned/native-mode input pipeline unchanged. Compare direct scalar stores against a 4x4 transpose across the four lane-K groups using four ds_bpermute operations, then one 16-byte row-major output store per lane. No output LDS exchange or CTA barrier. Vector stores fall back to four scalar stores for unaligned output base/row stride.

40 full-range INT8/FP64 combinations pass, including tails, ragged lengths, padded strides and full input/guard checks. Ten extra checks pass with C stride257 and A stride272 or9437184 (>2GiB span), covering the native and 64-bit-copy paths and scalar output-alignment fallback.

32 performance-output checks pass;3 HIP-event samples,warmup2/repeat5,median. This remains experimental; no production selector change.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|6048|53.294880|324.934|
|2048|6144|7168|96|7204|46.816383|369.898|
|2048|6144|7168|96|8402|44.763165|386.865|
|2048|6144|7168|96|8601|51.272949|337.747|
|2048|6144|7168|96|8602|45.154106|383.516|
|2048|6144|7168|96|8603|50.526875|342.735|
|2048|6144|7168|96|8604|45.506039|380.550|
|2048|6144|7168|96|8605|45.679577|379.104|
|2048|4096|8192|128|6048|55.521393|316.854|
|2048|4096|8192|128|7204|44.582268|394.601|
|2048|4096|8192|128|8402|42.772408|411.298|
|2048|4096|8192|128|8601|47.428596|370.919|
|2048|4096|8192|128|8602|42.690617|412.086|
|2048|4096|8192|128|8603|46.363861|379.437|
|2048|4096|8192|128|8604|42.079159|418.074|
|2048|4096|8192|128|8605|43.678612|402.764|
|16384|3072|4096|48|6048|65.861237|300.499|
|16384|3072|4096|48|7204|56.947826|347.532|
|16384|3072|4096|48|8402|53.239441|371.740|
|16384|3072|4096|48|8601|66.030708|299.727|
|16384|3072|4096|48|8602|55.517494|356.486|
|16384|3072|4096|48|8603|66.494453|297.637|
|16384|3072|4096|48|8604|55.638966|355.708|
|16384|3072|4096|48|8605|56.599571|349.671|
|16384|4096|6144|8|6048|21.982906|300.100|
|16384|4096|6144|8|7204|18.884954|349.329|
|16384|4096|6144|8|8402|18.154972|363.375|
|16384|4096|6144|8|8601|20.308027|324.850|
|16384|4096|6144|8|8602|17.951387|367.496|
|16384|4096|6144|8|8603|20.043674|329.135|
|16384|4096|6144|8|8604|17.927738|367.981|
|16384|4096|6144|8|8605|18.208315|362.311|

Scalar stores regress; the wave-transposed vector output gives only a modest large-M gain. Removing output staging is not sufficient to close the BLAS gap. Matched PMC comparison follows in iteration44; packed small/medium-M combination is screened separately in iteration43.

Library SHA256:5de720aab1bb18b675d07aa57cb5f24871ece00149c823765f8a2e0e8e4e5b1e
