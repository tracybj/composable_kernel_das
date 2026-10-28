# Iteration 34: medium-M packed deep-K tiles

gfx936 physical GPU7; INT8 NT -> INT32, no scale. Compare native direct-to-LDS ping-pong input buffers for medium-M output tiles. New problem/policy/pipeline/epilogue/kernel layers reuse native MMAC and exact INT32 semantics and remain explicit development candidates; production selector unchanged.

40 full-range INT8/FP64 oracle combinations pass, including padded strides, M/N/K tails and unequal group lengths, with full input copies and output guards. All 196 M32/64 target shapes (7 candidates for M32, 9 for M64) = 1556 output checks pass. Performance screening: three HIP-event samples, warmup3/repeat10, median; independent full-output reference for random [-8,8] inputs, guards and 256 sampled input elements per operand. This screening is not public-interface acceptance.

Oracle best coverage: 193/196 reach frozen targets; minimum R=0.943815, median R=1.386693. Residual misses remain in long-K, small-G shapes. A simple M/K/G dispatch also needs tuning; selecting the fastest candidate per benchmark shape is not a supported production solution.

D2L does not win on these shapes. Next compare packed ping-pong with one CTA barrier per K tile. See iteration_34_medium_m_audit.json and iteration_34_medium_m_best.csv. Source/build identity and all raw samples remain in results/medium_m_v34_screen and build/v34.
