# Iteration52: bind MMAC results to operand-stage ordering

Keep the exact native I8I8I32 fragment ABI and v50 A-dual-slot/B-direct structure. Add a typed MMAC policy: either volatile native v_mmac_i32_16x16x32_i8 with a memory boundary, or the CK-Tile native intrinsic followed by an accumulator-tied compiler boundary. This prevents later A-half LDS reads from moving ahead of completion of the earlier half. No arithmetic format or K coverage change.

66 full-range INT8/FP64 checks (60 boundary combinations plus six repeated smoke checks),32 performance-output records PASS. All six variants declare244 VGPRs,zero spills and zero private bytes. Three HIP-event samples,warmup2/repeat5,median; physical GPU7,gfx936,INT8 NT to INT32,no scale.

Best new-candidate throughput on the four representatives is approximately417-484 TOPS. The ordering change removes the verified spill problem and improves over the spilling B-bypass control, but does not consistently beat the previous split-role family or meet the frozen BLAS targets. Do not promote as default. Next matched counters compare9003,9301,9302 on the same allocations, separately from no-profiler performance.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|9003|36.371613|476.122|
|2048|6144|7168|96|9101|39.900894|434.008|
|2048|6144|7168|96|9301|37.972824|456.045|
|2048|6144|7168|96|9302|38.097496|454.552|
|2048|6144|7168|96|9303|48.306938|358.485|
|2048|6144|7168|96|9304|38.559353|449.108|
|2048|6144|7168|96|9305|38.025047|455.418|
|2048|6144|7168|96|9306|37.580891|460.801|
|2048|4096|8192|128|9003|36.596535|480.706|
|2048|4096|8192|128|9101|38.265530|459.740|
|2048|4096|8192|128|9301|36.476696|482.286|
|2048|4096|8192|128|9302|36.407417|483.203|
|2048|4096|8192|128|9303|47.489304|370.445|
|2048|4096|8192|128|9304|37.583992|468.077|
|2048|4096|8192|128|9305|36.335770|484.156|
|2048|4096|8192|128|9306|36.758904|478.583|
|16384|3072|4096|48|9003|46.694870|423.841|
|16384|3072|4096|48|9101|51.843224|381.751|
|16384|3072|4096|48|9301|47.510231|416.567|
|16384|3072|4096|48|9302|47.543385|416.277|
|16384|3072|4096|48|9303|58.148823|340.354|
|16384|3072|4096|48|9304|48.541718|407.715|
|16384|3072|4096|48|9305|47.569462|416.049|
|16384|3072|4096|48|9306|48.072662|411.694|
|16384|4096|6144|8|9003|14.473724|455.796|
|16384|4096|6144|8|9101|15.793149|417.717|
|16384|4096|6144|8|9301|14.849886|444.251|
|16384|4096|6144|8|9302|14.827261|444.928|
|16384|4096|6144|8|9303|18.646587|353.795|
|16384|4096|6144|8|9304|14.894493|442.920|
|16384|4096|6144|8|9305|14.881404|443.310|
|16384|4096|6144|8|9306|14.777885|446.415|

Library SHA256: 80a6fafb6c86fc2add4610b26bb211293285a11399cf200367581a8422114a36

All1178 target acceptance remains incomplete.
