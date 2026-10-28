# Iteration 33: medium-M packed deep-K tiles

gfx936 physical GPU7; INT8 NT -> INT32, no scale. Compare deeper BK512 with narrower BN32/64 and two-wave variants. These reuse the existing problem/policy/pipeline/kernel implementation and remain explicit development candidates; production selector unchanged.

36 full-range INT8/FP64 oracle combinations pass, including padded strides, M/N/K tails and unequal group lengths, with full input copies and output guards. All 196 M32/64 target shapes (7 candidates for M32, 10 for M64) = 1648 output checks pass. Performance screening: three HIP-event samples, warmup3/repeat10, median; independent full-output reference for random [-8,8] inputs, guards and 256 sampled input elements per operand. This screening is not public-interface acceptance.

Oracle best coverage: 193/196 reach frozen targets; minimum R=0.947629, median R=1.387757. Residual misses remain in long-K, small-G shapes. A simple M/K/G dispatch also needs tuning; selecting the fastest candidate per benchmark shape is not a supported production solution.

Next: test native D2L medium-M input buffers, then validate a dimension-based default rule. See iteration_33_medium_m_audit.json and iteration_33_medium_m_best.csv. Source/build identity and all raw samples remain in results/medium_m_v33_screen and build/v33.
