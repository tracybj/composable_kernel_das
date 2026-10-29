# Iteration76: M-group CTA access order

Compare M-group sizes1/2/4/5/8/16 with both internal orientations. Keep the current typed WASP pipeline, A batch4, prefetch, single publication/reuse barrier and full original mathematical workload. Twelve development instances only; no case-ID production dispatch and no public selector change.

228 full-range INT8/FP64 boundary checks and72 performance-output checks PASS. Each of six representative shapes exercises all12 candidates. Resource extraction contains12 new batch4 kernels with214VGPR plus one pre-existing public batch8 instantiation with232VGPR; all have no scratch/spill. Timing uses the probe's three HIP-event samples and median, warmup2/repeat5, physical GPU7.

| M | N | K | G | best development candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|256|1536|2048|4|21051|0.043488|148.143|144.476|1.0254|
|256|3072|4096|4|21021|0.085088|302.861|304.896|0.9933|
|16384|3072|4096|48|21020|36.423641|543.362|574.971|0.9450|
|16384|4096|6144|8|21050|13.138684|502.110|636.582|0.7888|
|2048|6144|7168|96|21021|33.369179|518.961|575.891|0.9011|
|2048|4096|8192|128|21020|28.047512|627.228|601.440|1.0429|

Changing CTA order alone does not close the large-M gap: the M16384,N4096,K6144,G8 representative remains about502TOPS against636.6TOPS. Some small-M representatives benefit, but this screening is insufficient to change a public dimension rule. Preserve the result rather than replacing the all1178 public measurement with oracle winners. Next test the independent power-of-two buffer-record stride decomposition observed in frozen BLAS ISA. Mathematical and performance acceptance remain separate.
