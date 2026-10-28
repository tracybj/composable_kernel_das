# Iteration44: matched generic/aligned/direct-output counters

gfx936 physical GPU7; INT8 NT -> INT32,no scale. Same A/B/C allocations and values across paths7204,8402,8602;2 warmups plus1 measured dispatch each. All three paths pass full-output reference, guards and full input clones. Nine dispatch counter rows are captured; normal performance ranking remains iteration42's separate no-profiler run.

The aligned path reduces reported VALU instructions from1,640,235,008 to1,414,004,736 and aggregate LDS-wait counter substantially, but no-profiler time gains only about4%. Wave-transposed no-LDS output further reduces VALU to1,408,499,712 and the reported bank-conflict counter to0; LDS instruction count increases slightly because ds_bpermute is counted. Cache hit/miss counters remain effectively unchanged. Thus eliminating the output conflict does not close the runtime gap and is not evidence that it was the dominant bottleneck.

All three paths reserve64KiB LDS, report204 architectural VGPRs and no scratch. Raw aggregate counts are not converted to occupancy or a stall-time percentage. Next compare wider global copy distribution and producer/consumer overlap. The initial filter produced an empty CSV despite valid computation; that terminal run is retained at results/profile_v44_compare. Corrected substring filtering captures all9 rows in fresh results/profile_v44b_compare.
