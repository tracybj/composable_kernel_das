# P6e cache expansion checkpoint — experiment only, not for merge

Full predeclared scope completed:212 K>4096 shapes plus15 new1179..1193 and4 short-K controls;231 shapes,1386 accepted logical jobs,3 adjacent original/new pairs each. Measured official benchmark SHA1052b95c681371b0713c357ade4e3ede659df08dd4d763e3cf6023e9ab5d0d2d, card4 only, warmup10/samples30/single launch/include-args-build0. CPU accuracy/guards each perf job; initial13 full-output accuracy jobs precede performance. No1193 full replay.

212 primary shapes: median paired change-1.788%;96 improve>2%,0 regress>2%, worst+0.241%. Descriptive within-round one-call totals (3-round medians) 1697.645924→1563.958232ms; median paired total change-7.875%. Frozen historical R>=0.99:125→163; BLAS oldcard0/repeat20 vsCKcard4/singlelaunch, not an equal-protocol claim. Labels:{'uncertain': 158, 'stable_improvement': 54}. Micro effects and variable rounds are uncertain.

|ID|Original ms|NoSLC ms|Median paired change|round epochs|
|---|---:|---:|---:|---|
|1178|56.512924|29.820374|-47.233%|[0, 0, 2]|
|1177|116.221350|59.501564|-48.793%|[0, 0, 2]|
|1126|19.617599|19.192223|-2.168%|[0, 0, 1]|
|1164|22.457211|21.821659|-2.816%|[0, 0, 2]|
|1095|0.428163|0.410641|-4.186%|[0, 0, 1]|

Time columns are measured medians. Changes are medians of within-epoch paired ratios; they need not equal independent time-column ratios. Initial paired profiler/ISA evidence remains in cache_checkpoint.md/cache_isa.json: only51 SLC bits change, original five shaders unchanged;1178 HBM40.306→11.595GB,L2 hit52.92→87.75%,latency695.6→371.7cycles. M01=1 is slower than retainedM01=8 on the three checked long-K shapes.

Recovery is part of the evidence. One MKFD gate at10:15:58, then an ordinal startup failure at10:49:33 with contemporaneoushymgr unregister/re-register10:49:06..57. Firstoldrunner resumed after only one retry (protocol gap); stopped own orchestration, allowed child to finish, retained all values.33 raw keys are excluded by recorded time windows/full affected pairs, never by speed. Two independent recoveries each passed two60s-spaced healthy gates and a previously validated exact-hash ID76base full-output probe. Epochs0/1/2 mark valid windows. Raw logs remain; accepted pairs never cross epoch. Pure logged waits180s count against the original14:53 deadline.

The new experimental entry supports a broader correctness domain than the tested performance policy. Recommendation here is the evaluatedK>4096 attribute scope, defaultoff, no per-shape best pick. The15newK512 shapes are diagnostic controls, not members of that policy. Small-M precision/performance and the final goal remain ongoing; this checkpoint is not completion of the8-hour task.
