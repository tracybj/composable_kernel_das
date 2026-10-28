# Iteration 28: native D2L, ping-pong and wave roles

gfx936 physical GPU7; INT8 NT -> INT32, no scale.

New code follows problem/policy/pipeline/epilogue/kernel split. Dense K64 LDS rows use an XOR of K-vector and row bits. Native CK dwordx4 D2L loads avoid packed input VGPR staging; 128-bit LDS reads feed two native MMACs using a shared K permutation. SRD bounds are capped below 2 GiB; very large strides use a 64-bit-addressed register-copy fallback into the same LDS layout.

Variants: 6701 single input buffer/normal epilogue; 6702 ping-pong/normal; 6703 ping-pong/merged epilogue; 6704 eight compute + four producer waves; 6705 producer version with M-tile groups8. Declared LDS allocation: single32 KiB; ping-pong64 KiB. These sizes are not occupancy measurements.

Twenty full-range INT8/FP64 checks pass, covering padded strides, M/N/K tails and unequal group lengths; full input copies and output guards verified. Extreme-stride fallback is not yet exhaustively tested. Twenty-eight development performance records pass full [-8,8] output/guard checks, 3 samples warmup2/repeat5. Production selector unchanged.

| M | N | K | G | candidate | ms | TOPS | CK/target |
|---:|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|6041|53.552574|323.370|0.5615|
|2048|6144|7168|96|6048|53.326103|324.744|0.5639|
|2048|6144|7168|96|6701|71.131287|243.456|0.4227|
|2048|6144|7168|96|6702|48.277916|358.700|0.6229|
|2048|6144|7168|96|6703|48.534676|356.803|0.6196|
|2048|6144|7168|96|6704|59.742290|289.867|0.5033|
|2048|6144|7168|96|6705|55.638203|311.249|0.5405|
|2048|4096|8192|128|6041|55.326427|317.971|0.5287|
|2048|4096|8192|128|6048|55.546619|316.710|0.5266|
|2048|4096|8192|128|6701|70.973740|247.869|0.4121|
|2048|4096|8192|128|6702|47.906967|367.216|0.6106|
|2048|4096|8192|128|6703|48.169495|365.214|0.6072|
|2048|4096|8192|128|6704|59.393810|296.196|0.4925|
|2048|4096|8192|128|6705|57.090004|308.148|0.5124|
|16384|3072|4096|48|6041|98.634567|200.652|0.3490|
|16384|3072|4096|48|6048|65.881546|300.406|0.5225|
|16384|3072|4096|48|6701|84.420746|234.435|0.4077|
|16384|3072|4096|48|6702|58.956787|335.690|0.5838|
|16384|3072|4096|48|6703|58.154064|340.324|0.5919|
|16384|3072|4096|48|6704|71.159409|278.125|0.4837|
|16384|3072|4096|48|6705|68.247414|289.992|0.5044|
|16384|4096|6144|8|6041|32.552090|202.662|0.3184|
|16384|4096|6144|8|6048|21.968763|300.293|0.4717|
|16384|4096|6144|8|6701|28.999960|227.485|0.3574|
|16384|4096|6144|8|6702|20.103643|328.153|0.5155|
|16384|4096|6144|8|6703|20.305243|324.895|0.5104|
|16384|4096|6144|8|6704|24.405275|270.313|0.4246|
|16384|4096|6144|8|6705|22.400284|294.508|0.4626|

Result: ping-pong improves best representative throughput to about328-367 TOPS, still below575-637 target TOPS. Single-buffer D2L regresses. Dedicated producer waves regress in this configuration. Do not promote based on these four samples. Next isolate LDS mapping and cache/scheduling factors; raw vendor ISA is not included.

Build command is preserved in experiments/direct_lds_v28/build_command.json. Library SHA256: 58b50d78887f82f16ad3d075c8471be532dbfdf20cc43ad1b3e9e7f7d936849f
