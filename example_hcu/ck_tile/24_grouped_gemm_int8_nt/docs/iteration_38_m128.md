# Iteration 38: M128 coverage screen

gfx936 physical GPU7, INT8 NT -> INT32, no scale. Reuse frozen build/v34/libprobe.so to compare the existing selector's candidate against masked128x128, packed deep-K and medium-D2L variants. No production code change in this iteration.

All78 M128 shapes and 620 candidate records pass independent full-output [-8,8]/FP32 integer reference, output guards and256 input samples. Three HIP-event samples, warmup3/repeat10, median. Candidate best coverage76/78, minR=0.966455, medianR=1.242457. Two cases remain below target; this is an oracle screen, not public default acceptance.

Hypothesis for the small-G residual: narrow N tiles can increase output CTA count while retaining M128 weight reuse. Next test BN32/64 with BK128/256. Raw samples and library identity remain in results/m128_v38_screen; see iteration_38_m128_audit.json and iteration_38_m128_best.csv.
