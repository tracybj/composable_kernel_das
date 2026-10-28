# Iteration 37: default medium-M deep-K dispatch

Register five masked packed instances: M32/BN64/BK128 (7501), M32/BN64/BK256 (7502), M32/BN128/BK256 (7504), M64/BN128/BK128 (7603), M64/BN128/BK256 (7604). All reuse the current CK-Tile problem/policy/pipeline/epilogue/kernel layers; no new arithmetic format or scale semantics. Total INT8 instances23; total registry43 including20 FP entries.

Only common, nonempty, nonragged requests with 16<M<=64 gain these paths. For M<=32: K<1024 or G<=4 selects7501, otherwise G<=32 selects7502 and larger G selects7504. For 32<M<=64: K<2048 or G<=4 selects7603, otherwise7604. These are operator resource thresholds, with no benchmark IDs, labels or phase input. M<=16 and M>64 selection stays unchanged; existing architecture, layout and capacity guards remain.

Fresh public library builds. Original80 and INT8 30 host selector checks pass; all1178 dimension-only replay/relabel/reverse tests pass. 28 full-range INT8/FP64 public boundaries pass, including new M17/32/33/64, K16 tails, N4 tails, padded strides, nondefault stream, full input clones, output guards and error contracts. These are numerical/ABI checks, not an all1178 throughput claim.

Next run default select/query/run on all196 M32/64 target shapes with three HIP-event samples, warmup10/repeat50, median. Known long-K small-G gaps remain to be resolved; performance results will be recorded separately. Scope remains INT8 NT -> INT32, no scale, gfx936 physical GPU7.

## Completed public validation

196/196 numerical checks pass; 192/196 reach fixed spreadsheet targets, minimum R=0.944205, median R=1.385772. Four long-K small-G rows remain below target and are included unmodified in iteration_37_public_medium_m.csv. This is not full1178 completion.
