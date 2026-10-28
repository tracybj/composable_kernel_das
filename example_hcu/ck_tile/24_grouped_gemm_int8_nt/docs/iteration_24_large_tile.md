# Iteration 24: existing large-M tile/schedule comparison

2026-10-28; gfx936 physical GPU7; INT8 NT -> INT32, no scale.

Compare already compiled instances 6041 (256x256 M-first), 6042 (128x256), 6044 (128x128 masked), 6048 (256x256 N-first). All sampled M/N/K are aligned to the tested unmasked domains. This tests whether dispatch alone can close the large-M gap before changing the dataflow.

Frozen probe binary SHA256: 27365451e032e529519c8359d9d02dac1c2f3833b35ba658101fd9960f0d63ab

Three samples, warmup2/repeat5. Full independent output and guards checked, [-8,8] inputs. Screening only.

| M | N | K | G | candidate | ms | TOPS | CK/target |
|---:|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|6041|53.557983|323.338|0.561|
|2048|6144|7168|96|6042|57.798939|299.613|0.520|
|2048|6144|7168|96|6044|78.378006|220.946|0.384|
|2048|6144|7168|96|6048|53.340668|324.655|0.564|
|2048|4096|8192|128|6041|55.378620|317.671|0.528|
|2048|4096|8192|128|6042|60.621593|290.197|0.483|
|2048|4096|8192|128|6044|86.575409|203.201|0.338|
|2048|4096|8192|128|6048|55.621300|316.285|0.526|
|16384|3072|4096|48|6041|100.886284|196.173|0.341|
|16384|3072|4096|48|6042|68.872116|287.362|0.500|
|16384|3072|4096|48|6044|116.279526|170.204|0.296|
|16384|3072|4096|48|6048|65.912949|300.263|0.522|
|16384|4096|6144|8|6041|31.303577|210.745|0.331|
|16384|4096|6144|8|6042|22.982235|287.051|0.451|
|16384|4096|6144|8|6044|39.379288|167.526|0.263|
|16384|4096|6144|8|6048|22.038395|299.344|0.470|

All 16 checks pass. Existing variants remain around 300-325 TOPS at best on these cases and do not close the spreadsheet gap. N-first substantially beats M-first for M16384; smaller tiles alone do not win. Keep current large-M dispatch. Next compare packed LDS/read scheduling and MMAC operand supply; do not interpret these results as a hardware limit.

Controller and raw records: results/large_tile_v24. No raw vendor ISA is published.
