# Iteration105:smallM64 producer/consumer geometry

Hypothesis:the legacy M64 route does not exploit static INT8 packed weights,and M128 would waste row work. Added separate CK-Tile Problem/Pipeline/Kernel/Epilogue,64x(64/128/256)x128 tiles,one A producer wave and2/4/8 consumers. Each wave owns64x32 output with32INT32 accumulators. A uses two8KiB slots(total16KiB LDS),same established publication/reuse barriers. B retains format1 and direct MMAC fragments;activations remain raw. No FP16/BF16 interface changes.

CPU A-address model PASS.108 full-rangeINT8/FP64 checks PASS,including native and odd K-stage counts,padding,ragged/empty groups,tails,output guards,and full input immutability. All9 kernels use86VGPR,no spill,16KiB LDS. All92 original M64 requirements,884 same-build measurements PASS,2warmups/5repeats/3sample median and rotated order. Four requirements withN1232 orK1232 cannot use native candidates and retain legacy output checks.

The broad N64/A2 packed candidate has minimum speedup0.9964 and median1.1613 on88 native shapes. N256 can regress on other shapes,so it is restricted by dimensions toN>=4096,K>=4096,G<=12;remaining native M64 usesN64/A2. This dimension-only draft reaches all92 targets in the development screen. It is calibrated on this screen and must pass independent public-API acceptance before production claims.

Previously failing M64/N4096/K6144/G8:108.50 to148.39TOPS,target114.48. M64/N4096/K7168/G8:110.38 to149.84TOPS,target116.60. These are new-run comparisons,not replacements of v103's frozen records. Main public API remains v103 until integration gates and full1178 measurements finish. Next integrate additive routes8/9 using unchanged packed format and ownership/lifetime contract.
