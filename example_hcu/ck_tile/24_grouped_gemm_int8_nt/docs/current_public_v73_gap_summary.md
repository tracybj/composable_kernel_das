# Current public-v73 fixed-target gaps

All1178 actual default-API records pass the benchmark reference/guard checks.915 reach the frozen target;263 remain below it:14 at M<256 and249 at M>=256. Every row has R>=0.7617. Passing numerical checks and being above50% do not complete the current all1178 fixed-BLAS-target objective.

| M | Total | Reaches target | Remaining | Median R | Minimum R |
|---:|---:|---:|---:|---:|---:|
|1|112|112|0|1.3697|1.0348|
|2|112|112|0|1.3709|1.0352|
|4|112|112|0|1.3743|1.0371|
|8|112|112|0|1.3687|1.0358|
|16|112|112|0|1.3388|1.0072|
|32|104|102|2|1.3635|0.9747|
|64|92|90|2|1.4196|0.9443|
|128|78|68|10|1.1995|0.8670|
|256|38|24|14|1.0741|0.9218|
|2048|108|37|71|0.9400|0.7901|
|4096|88|30|58|0.9534|0.7865|
|8192|72|4|68|0.9394|0.7836|
|16384|38|0|38|0.8972|0.7617|

Remaining gaps are in current_public_v73_remaining.csv,including original dimensions,phase,actual public instance,ms,TOPS and frozen target. These are actual default results; the earlier per-shape oracle is a separate experiment. Next test structured-buffer cache-swizzle eligibility on non-power-of-two pitches,with unchanged controls and full-range checks. Medium-M14 misses remain in scope and require separate instance/dispatch work.
