# Iteration 31: medium-M packed deep-K tiles

gfx936 physical GPU7; INT8 NT -> INT32, no scale. Add masked, peeled packed-load instances for M32/64 with BK128/256 and BN64/128/256, varying wave geometry. These reuse the existing problem/policy/pipeline/kernel implementation and remain explicit development candidates; production selector unchanged.

44 full-range INT8/FP64 oracle combinations pass, including padded strides, M/N/K tails and unequal group lengths, with full input copies and output guards. All 196 M32/64 target shapes x 8 candidates = 1568 output checks pass. Performance screening: three HIP-event samples, warmup3/repeat10, median; independent full-output reference for random [-8,8] inputs, guards and 256 sampled input elements per operand. This screening is not public-interface acceptance.

Oracle best coverage: 192/196 reach frozen targets; minimum R=0.943941, median R=1.384742. Four misses cluster in long-K, G4/8 shapes. A simple M/K/G dispatch also needs tuning; selecting the fastest candidate per benchmark shape is not a supported production solution.

Next: increase reduction depth/narrow BN for long-K small-G and compare two-wave configurations, then validate a dimension-based default rule. See iteration_31_medium_m_audit.json and iteration_31_medium_m_best.csv. Source/build identity and all raw samples remain in results/medium_m_v31_screen and build/v31.
