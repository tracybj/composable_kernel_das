# Iteration55: hoisted copy offsets and bulk m0 updates

Precompute the eight A-copy vector offsets and wave LDS base, issue eight raw-buffer direct-to-LDS loads in one typed policy, advance m0 by4096 between interleaved panels. Compare constant descriptor plus kb in vector offsets with scalar64-bit descriptor-base advancement. The base advancement uses unsigned carry arithmetic; native aligned/tail and generic fallback bounds are retained. Compare both original and equivalent-transposed internal mappings.

88 full-range INT8/FP64 checks (80 boundary combinations plus eight repeated smoke checks),40 representative performance-output checks PASS. Eight new kernels use243-244 VGPRs,no scratch/spills. Three HIP-event samples,warmup2/repeat5,median,physical GPU7.

Gains are small and shape-dependent; no target completion or default promotion. C++ source-level hoisting alone does not prove that the backend preserves those values rather than rematerializing them. The subsequent WASP-source inspection identifies a structured B/cache-swizzle path and explicit opaque invariant operands, to be tested with cache on/off ablations.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|9003|36.360191|476.271|
|2048|6144|7168|96|9401|36.803261|470.537|
|2048|6144|7168|96|9501|36.711449|471.714|
|2048|6144|7168|96|9502|37.246590|464.937|
|2048|6144|7168|96|9503|36.864220|469.759|
|2048|6144|7168|96|9504|38.261402|452.605|
|2048|6144|7168|96|9505|36.939545|468.801|
|2048|6144|7168|96|9506|38.184280|453.519|
|2048|6144|7168|96|9507|37.566971|460.972|
|2048|6144|7168|96|9508|38.708057|447.383|
|2048|4096|8192|128|9003|36.544346|481.393|
|2048|4096|8192|128|9401|36.158585|486.529|
|2048|4096|8192|128|9501|36.281658|484.878|
|2048|4096|8192|128|9502|36.507992|481.872|
|2048|4096|8192|128|9503|36.452248|482.609|
|2048|4096|8192|128|9504|37.948025|463.586|
|2048|4096|8192|128|9505|36.181595|486.219|
|2048|4096|8192|128|9506|37.900185|464.172|
|2048|4096|8192|128|9507|36.526009|481.635|
|2048|4096|8192|128|9508|37.626427|467.549|
|16384|3072|4096|48|9003|46.851063|422.428|
|16384|3072|4096|48|9401|49.229652|402.018|
|16384|3072|4096|48|9501|49.129078|402.841|
|16384|3072|4096|48|9502|59.760658|331.175|
|16384|3072|4096|48|9503|49.165718|402.541|
|16384|3072|4096|48|9504|49.668404|398.467|
|16384|3072|4096|48|9505|49.341011|401.111|
|16384|3072|4096|48|9506|49.525875|399.614|
|16384|3072|4096|48|9507|47.263378|418.743|
|16384|3072|4096|48|9508|48.694836|406.433|
|16384|4096|6144|8|9003|14.561885|453.037|
|16384|4096|6144|8|9401|15.370813|429.195|
|16384|4096|6144|8|9501|15.163229|435.070|
|16384|4096|6144|8|9502|14.501757|454.915|
|16384|4096|6144|8|9503|15.138525|435.780|
|16384|4096|6144|8|9504|15.542781|424.446|
|16384|4096|6144|8|9505|15.337596|430.124|
|16384|4096|6144|8|9506|15.519709|425.077|
|16384|4096|6144|8|9507|14.650621|450.293|
|16384|4096|6144|8|9508|14.967964|440.746|

Library SHA256: 222b911c51f3f1f98f0389a34bbb229739a46f1c7939388b0a49f346e110186b

All1178 fixed-target acceptance remains incomplete.
