# Iteration48: separate producer and consumer loop lifetimes

Physical GPU7, gfx936; INT8 NT to exact INT32, no scale. Split wave-uniform producer/consumer branches into separate loop functions. Only producers hold global-prefetch tiles; only consumers hold MMAC accumulators. Match the first publication and retire/publish barrier phases, then use the previously validated no-LDS output epilogue. Padded K128 is included in the native signed-offset proof; large strides use the generic fallback.

54 full-range FP64 checks PASS (48 boundary combinations plus six repeated aligned smoke checks). All36 performance-output records PASS. Timing is three HIP-event samples with warmup2/repeat5, median; these four representative shapes are screening only.

The six new variants use233-234 VGPRs, no scratch and no spills, compared with256 VGPRs and seven spills for the prior paired producer variant. This confirms the lifetime change removed spills, but does not establish that spills explain the entire BLAS gap.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|8602|45.163609|383.435|
|2048|6144|7168|96|8901|41.455517|417.732|
|2048|6144|7168|96|8902|64.811676|267.194|
|2048|6144|7168|96|9001|36.493660|474.529|
|2048|6144|7168|96|9002|36.822170|470.296|
|2048|6144|7168|96|9003|36.339771|476.539|
|2048|6144|7168|96|9004|35.922779|482.070|
|2048|6144|7168|96|9005|36.568699|473.555|
|2048|6144|7168|96|9006|36.609818|473.024|
|2048|4096|8192|128|8602|42.900661|410.068|
|2048|4096|8192|128|8901|42.943260|409.661|
|2048|4096|8192|128|8902|64.010422|274.833|
|2048|4096|8192|128|9001|38.473305|457.257|
|2048|4096|8192|128|9002|38.952347|451.634|
|2048|4096|8192|128|9003|36.577305|480.959|
|2048|4096|8192|128|9004|37.570938|468.239|
|2048|4096|8192|128|9005|38.471832|457.274|
|2048|4096|8192|128|9006|38.795963|453.454|
|16384|3072|4096|48|8602|55.537140|356.360|
|16384|3072|4096|48|8901|52.006744|380.551|
|16384|3072|4096|48|8902|70.434128|280.989|
|16384|3072|4096|48|9001|47.239605|418.954|
|16384|3072|4096|48|9002|47.729141|414.657|
|16384|3072|4096|48|9003|46.815224|422.752|
|16384|3072|4096|48|9004|46.567223|425.003|
|16384|3072|4096|48|9005|47.758999|414.397|
|16384|3072|4096|48|9006|47.252052|418.843|
|16384|4096|6144|8|8602|17.896988|368.613|
|16384|4096|6144|8|8901|17.029629|387.388|
|16384|4096|6144|8|8902|24.443804|269.887|
|16384|4096|6144|8|9001|15.644221|421.694|
|16384|4096|6144|8|9002|15.591580|423.117|
|16384|4096|6144|8|9003|14.526013|454.156|
|16384|4096|6144|8|9004|15.290556|431.447|
|16384|4096|6144|8|9005|15.647740|421.599|
|16384|4096|6144|8|9006|15.613917|422.512|

No public default change. Full1178 target acceptance remains incomplete. Library SHA256: 6f5f10d0e7557aab67a700fbfb473999d7d71a6fb1bfb8ed61cf47bade31c081
