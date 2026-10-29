# Iteration83:512x256x64 producer/consumer structure

## Structure and contract

New typed problem,pipeline,epilogue and kernel. Tile512x256,Kstage64;four producers load128 rows each,eight consumer waves own512 rows by32 columns each. Two512x64 A stages keep LDS at64KB. Producer rows are swizzled in four16-byte chunks;consumer A addresses use row*64 plus the matching xor chunk. B uses one64-byte logical K half per stage,record0/1024 variants,both internal orientations,A read batches2/4. Original physical A/B/C,independent group weights,INT32 exact accumulation and no-scale semantics remain unchanged. Unsupported geometry/strides execute two masked256-row packed fallbacks per CTA;only the internal transpose changes output indexing.

The ideal tile-volume model reduces B tile loads by half at equal full work while keeping A tile loads equal. It also doubles per-consumer accumulators from128 to256 INT32 values,reduces CTA count by half and doubles K-loop count. This is a structural tradeoff,not a promised gain.

## Validation

CPU address ownership/swizzle/record decomposition/output coverage checks PASS;these are not GPU validation.216 full-range INT8/FP64 checks PASS,including odd/even native stage counts(K320/448/576),both mappings,padded inputs,ragged lengths,tails and fallback. All56 performance outputs/guards PASS,seven original shapes,eight candidates. Timing uses GPU7,three HIP-event samples,warmup2/repeat5,median. Actual source/library identity and resources are preserved.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|24012|5.792127|47.457|201.009|0.2361|
|4096|1024|2048|64|24012|18.868475|58.272|374.064|0.1558|
|2048|1536|2048|8|24002|1.791391|57.541|414.146|0.1389|
|16384|3072|4096|48|24102|344.473053|57.454|574.971|0.0999|
|16384|4096|6144|8|24002|109.410622|60.296|636.582|0.0947|
|2048|6144|7168|96|24002|284.004639|60.975|575.891|0.1059|
|2048|4096|8192|128|24012|290.624817|60.532|601.440|0.1006|

All eight builds report private/scratch292--452 bytes per work item and nonzero VGPR/SGPR spill metadata,with declared255--256 VGPRs. Performance collapses to roughly34--61TOPS;do not promote this producer/consumer structure. The spill metadata is direct code-generation evidence;the measurements do not isolate every part of the slowdown.

DTK llc --help-hidden confirms support-512-vgprs and support-768-vgprs modes exist;this build already uses the established support-768 flag. Therefore the observed256-VGPR allocation/spills are not presented as proof of a universal hardware register ceiling. CK make_kernel<Threads,1> instantiates __launch_bounds__(Threads,1) when enabled,so workgroup shape and allocation constraints remain relevant.

Next test the same512x256x64 mathematical tile with eight cooperative waves and512 threads,removing the four dedicated producer waves. Each wave copies64 A rows using four raw16-byte loads per lane,prefetches the next A/B stage before current MMAC,waits and publishes/reuses at the stage boundary. This changes wave allocation and critical-path ownership;it may reduce spills but adds copy work to compute waves. Validate the whole compatible structure rather than treating this failed twelve-wave build as rejection of all larger-M tiles. Public v80 and its975/1178 formal target count remain unchanged;203 misses remain active.
