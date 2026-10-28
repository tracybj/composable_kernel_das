# Iteration 35: medium-M packed deep-K tiles

gfx936 physical GPU7; INT8 NT -> INT32, no scale. Compare packed global prefetch with two LDS buffers, reducing the steady-state loop to one CTA barrier per K tile. This increases LDS allocation while retaining exact integer math. These reuse the existing problem/policy/pipeline/kernel implementation and remain explicit development candidates; production selector unchanged.

32 full-range INT8/FP64 oracle combinations pass, including padded strides, M/N/K tails and unequal group lengths, with full input copies and output guards. All 196 M32/64 target shapes (7 candidates for M32, 6 for M64) = 1280 output checks pass. Performance screening: three HIP-event samples, warmup3/repeat10, median; independent full-output reference for random [-8,8] inputs, guards and 256 sampled input elements per operand. This screening is not public-interface acceptance.

Oracle best coverage: 192/196 reach frozen targets; minimum R=0.944194, median R=1.387816. Residual misses remain in long-K, small-G shapes. A simple M/K/G dispatch also needs tuning; selecting the fastest candidate per benchmark shape is not a supported production solution.

Most shapes still favor the single-buffer deep-K candidates. Next use hardware profiling to guide the remaining work. See iteration_35_medium_m_audit.json and iteration_35_medium_m_best.csv. Source/build identity and all raw samples remain in results/medium_m_v35_screen and build/v35.
