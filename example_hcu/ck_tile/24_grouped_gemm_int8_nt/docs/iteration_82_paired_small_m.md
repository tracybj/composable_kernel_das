# Iteration82: paired small-M v73/v80 recheck

The v80 formal all1178 snapshot has two shapes that were above the frozen target in v73 and below it in v80. Preserve both formal snapshots. Compare the two exact public libraries on identical A/B/C physical allocations,reverse version order in the middle of three samples,warmup10/repeat50,physical GPU7,INT8 NT to INT32 without scale. Both versions select the same existing small-M instance for each shape.

All12 full-output checks PASS,including output guards and full input clones. Inputs[-8,8],TF32-disabled exact FP32 reference. This is a separate timing diagnostic,not a replacement full1178 run or full-range numeric claim.

| ID (analysis join only) | Version | Median ms | TOPS | R |
|---:|---:|---:|---:|---:|
|508|73|0.486058|37.278|1.377943|
|508|80|0.485734|37.303|1.378860|
|1022|73|0.205786|62.613|1.007689|
|1022|80|0.205702|62.639|1.008097|

M16,N5120,K2304,G48 returns0.486ms in both versions,near the v73 formal0.486ms rather than the v80 formal0.736ms. M32,N4096,K6144,G8 returns about0.206ms in both versions,slightly above its fixed target throughput. The apparent small-M regressions do not reproduce in this paired run;the exact cause of the earlier time variation is not established. Keep the original v80 values and975/1178 formal count intact,with this independent diagnostic visible. No selector,algorithm,target or tolerance change. All1178-target optimization remains active.
