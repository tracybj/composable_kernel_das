# Iteration 39: aligned K-loop and address-mode hoisting

gfx936 physical GPU7; INT8 NT -> INT32, no scale. Start from the K64 native D2L ping-pong layout and remove row/K masks for M/N multiples256 and K multiple64. Candidate8401 keeps native-versus-64-bit-copy decisions within Stage;8402 hoists the stride-span guard outside the K loop and specializes the native-only loop.8403 uses sixteen compute waves,8404 swaps wave axes,8405 uses M tile group8. Generic shape/tail and >2GiB stride-span fallback remain. No default change.

40 full-range INT8/FP64 combinations pass, including aligned and ragged/tail layouts, padded strides, full input clones and guards. Five extra aligned256x256x256 checks with A stride9437184 bytes (span2406482176 bytes) pass through the 64-bit fallback and preserve the entire input allocation and output guards.

28 development performance-output records pass:3 HIP-event samples, warmup2/repeat5, median.

| M | N | K | G | candidate | ms | TOPS |
|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|6048|53.292053|324.951|
|2048|6144|7168|96|7204|46.793373|370.080|
|2048|6144|7168|96|8401|45.667900|379.201|
|2048|6144|7168|96|8402|44.987263|384.938|
|2048|6144|7168|96|8403|44.049789|393.130|
|2048|6144|7168|96|8404|46.847160|369.655|
|2048|6144|7168|96|8405|45.000187|384.827|
|2048|4096|8192|128|6048|55.602459|316.392|
|2048|4096|8192|128|7204|44.451546|395.761|
|2048|4096|8192|128|8401|43.478008|404.623|
|2048|4096|8192|128|8402|42.745434|411.557|
|2048|4096|8192|128|8403|43.528088|404.157|
|2048|4096|8192|128|8404|46.143387|381.250|
|2048|4096|8192|128|8405|43.572887|403.742|
|16384|3072|4096|48|6048|65.872536|300.447|
|16384|3072|4096|48|7204|56.999836|347.215|
|16384|3072|4096|48|8401|55.326103|357.719|
|16384|3072|4096|48|8402|53.576214|369.403|
|16384|3072|4096|48|8403|54.947540|360.184|
|16384|3072|4096|48|8404|55.515984|356.496|
|16384|3072|4096|48|8405|54.655479|362.108|
|16384|4096|6144|8|6048|22.032797|299.420|
|16384|4096|6144|8|7204|18.892029|349.199|
|16384|4096|6144|8|8401|18.487484|356.840|
|16384|4096|6144|8|8402|18.147772|363.520|
|16384|4096|6144|8|8403|18.500446|356.590|
|16384|4096|6144|8|8404|19.399355|340.066|
|16384|4096|6144|8|8405|18.049341|365.502|

Paired candidate8402 improves roughly4.0-6.4% over7204 across these four samples, reaching up to411.56 TOPS. Still below frozen targets. This disproves that merely removing masks/address-mode branches closes the remaining gap. Next compare no-LDS output paths and collect matched dynamic counters.

Library SHA256:7b353ffae889c61961baf821923ef3fc8eb495e4afca2684763eb6c04a141636
