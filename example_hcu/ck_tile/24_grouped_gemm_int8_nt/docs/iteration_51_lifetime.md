# Iteration51: compiler memory boundaries and volatile LDS reads

Keep the v50 A-only dual-slot/B-direct structure. Add compiler memory clobbers at K64-half boundaries, and compare volatile vector LDS reads. Both versions retain typed native MMAC intrinsic arithmetic.

66 full-range INT8/FP64 checks (60 boundary combinations plus six repeated smoke checks),32 performance-output records PASS. Same physical GPU7 and original dimensions, INT8 NT to INT32,no scale; three HIP-event samples,warmup2/repeat5,median.

Compiler memory clobbers alone do not solve the lifetime issue: metadata still declares256 VGPRs,24-28 spills,100-116 private bytes. Volatile LDS reads regress performance substantially. Native MMAC is a pure compute intrinsic, so a memory-only compiler boundary does not bind the updated accumulator values to the stage boundary. Next compare an ordered native-instruction policy with a native-intrinsic policy whose resulting fragment is tied to a compiler boundary. No default promotion.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|9003|36.367168|476.180|
|2048|6144|7168|96|9101|39.919964|433.801|
|2048|6144|7168|96|9201|39.834812|434.728|
|2048|6144|7168|96|9202|78.625946|220.249|
|2048|6144|7168|96|9203|49.690426|348.504|
|2048|6144|7168|96|9204|39.675705|436.471|
|2048|6144|7168|96|9205|39.806648|435.036|
|2048|6144|7168|96|9206|39.550907|437.849|
|2048|4096|8192|128|9003|36.582203|480.895|
|2048|4096|8192|128|9101|38.275513|459.620|
|2048|4096|8192|128|9201|38.243866|460.000|
|2048|4096|8192|128|9202|77.865234|225.931|
|2048|4096|8192|128|9203|49.566650|354.920|
|2048|4096|8192|128|9204|38.785210|453.580|
|2048|4096|8192|128|9205|38.082554|461.949|
|2048|4096|8192|128|9206|38.637688|455.312|
|16384|3072|4096|48|9003|46.763924|423.215|
|16384|3072|4096|48|9101|51.986931|380.696|
|16384|3072|4096|48|9201|51.828178|381.862|
|16384|3072|4096|48|9202|95.491211|207.257|
|16384|3072|4096|48|9203|63.098003|313.658|
|16384|3072|4096|48|9204|51.563152|383.825|
|16384|3072|4096|48|9205|51.967640|380.837|
|16384|3072|4096|48|9206|52.600311|376.257|
|16384|4096|6144|8|9003|14.505373|454.802|
|16384|4096|6144|8|9101|15.796221|417.636|
|16384|4096|6144|8|9201|15.833468|416.653|
|16384|4096|6144|8|9202|30.506842|216.249|
|16384|4096|6144|8|9203|19.686331|335.109|
|16384|4096|6144|8|9204|15.503197|425.530|
|16384|4096|6144|8|9205|15.774076|418.222|
|16384|4096|6144|8|9206|15.750941|418.837|

Library SHA256: e9d655074c8cc3f30271f83b99127e20b007ecde32e6dfe0fd04b8be117b1316

All1178 target acceptance remains incomplete.
