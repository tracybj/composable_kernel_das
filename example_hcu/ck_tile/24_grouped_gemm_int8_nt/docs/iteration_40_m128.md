# Iteration 40: M128 narrow-N deep-K candidates

gfx936 physical GPU7, INT8 NT -> INT32, no scale. Compare masked M128 output blocks with BN32/64, BK128/256, wave2x2/4x1, plus a 4x1 M128/BN128/BK128 variant. All reuse existing typed CK-Tile layers; production default unchanged.

24 full-range INT8/FP64 combinations pass, covering tails, unequal group lengths, padded strides, full input clones and output guards. All78 M128 shapes and 620 candidate records pass independent full-output [-8,8]/FP32 integer reference, guards and256 sampled input elements. Three HIP-event samples, warmup3/repeat10, median.

Oracle coverage77/78, minR=0.971636, medianR=1.244601. Narrow BN64/BK128 helps the G12 long-K residual; the G4,M128,N3072,K4096 case remains below target. Most cases still prefer existing128x128 or128x256 tiles. No benchmark case ID dispatch is introduced, and these oracle choices have not been integrated into a default rule.

Raw records: results/m128_v40_screen. Build command: experiments/m128_narrow_v40/build_command.json. Library SHA256:7bfbb02e9cedad1b038aa92d7221a0dd31909dac89bd42f3d5eed6d064648ed2. Further work: generic aligned-loop specialization based on dimensions and stride guards, plus residual small-G latency analysis.
