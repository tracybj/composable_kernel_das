# Iteration50: A-only LDS double buffer and direct B registers

BLAS ISA inspection shows buffer_load_dwordx4 results used directly as B operands in MMAC, with16 A fragments and2 B fragments per wave. New typed problem/policy/pipeline/kernel:256x256 output, eight compute waves laid out1x8, four A-copy waves, two32KiB A slots, consumer B K128 register tiles. A native async copies overlap current computation; one common publication/retirement barrier per K128 iteration. B follows the same K64-pair permutation as A, preserving exact integer arithmetic. Padded-K signed-offset guards retain generic fallback for tails/large strides.

66 full-range INT8/FP64 checks (60 boundary combinations plus6 repeated smoke checks) and32 representative performance-output records PASS. No default promotion. Three HIP-event samples,warmup2/repeat5,median.

Resource metadata:256 VGPRs with24-28 spills and100-116 private bytes. Own ISA confirms28 scratch-descriptor stores at the final compute, where both64-byte K halves of the A fragment set have been loaded before MMAC; the scheduler fence alone did not limit their simultaneous lifetime. Next test a compiler memory clobber combined with scheduler fences and volatile LDS reads. This negative result does not reject B bypass as a structure.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|9003|36.354172|476.350|
|2048|6144|7168|96|9004|35.937855|481.868|
|2048|6144|7168|96|9101|39.840252|434.669|
|2048|6144|7168|96|9102|39.546238|437.900|
|2048|6144|7168|96|9103|49.696762|348.459|
|2048|6144|7168|96|9104|39.691032|436.303|
|2048|6144|7168|96|9105|39.821529|434.873|
|2048|6144|7168|96|9106|48.998940|353.422|
|2048|4096|8192|128|9003|36.571739|481.032|
|2048|4096|8192|128|9004|37.895836|464.225|
|2048|4096|8192|128|9101|38.205208|460.466|
|2048|4096|8192|128|9102|38.684856|454.756|
|2048|4096|8192|128|9103|49.521782|355.241|
|2048|4096|8192|128|9104|38.793274|453.485|
|2048|4096|8192|128|9105|38.079769|461.982|
|2048|4096|8192|128|9106|48.739666|360.942|
|16384|3072|4096|48|9003|46.801430|422.876|
|16384|3072|4096|48|9004|46.927414|421.741|
|16384|3072|4096|48|9101|51.880852|381.474|
|16384|3072|4096|48|9102|52.565174|376.508|
|16384|3072|4096|48|9103|62.933868|314.476|
|16384|3072|4096|48|9104|51.502800|384.274|
|16384|3072|4096|48|9105|51.999157|380.606|
|16384|3072|4096|48|9106|61.184593|323.467|
|16384|4096|6144|8|9003|14.550428|453.394|
|16384|4096|6144|8|9004|15.223966|433.335|
|16384|4096|6144|8|9101|15.836540|416.573|
|16384|4096|6144|8|9102|15.761053|418.568|
|16384|4096|6144|8|9103|19.748636|334.052|
|16384|4096|6144|8|9104|15.503613|425.518|
|16384|4096|6144|8|9105|15.804029|417.430|
|16384|4096|6144|8|9106|19.108379|345.245|

Library SHA256: fcb12a9a13b3b71f1916370315f3cd9e40b2959d4a3ee845a6ed5a6b6137ba69

Full1178 fixed-target completion remains unproven.
