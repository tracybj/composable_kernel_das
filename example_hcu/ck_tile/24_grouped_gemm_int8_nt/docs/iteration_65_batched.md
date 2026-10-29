# Iteration64: complete short-live INT8 WASP kernel

Parameterize the uniform-C consumer by AReadBatch=1/2/4/8/16. Read each batch into local fragments, wait for LDS, then issue the ordered native I8 MMAC operations. Keep current WASP double-buffer A direct-to-LDS, structured B loads, quarter-stage producer/consumer barriers and physical external NT layout. Ten candidates cover five A batches and original/equivalent-transposed internal mapping with M CTA group4. Retain generic masked fallback for unsupported/tail cases. Selection IDs are development instance IDs, not workload case dispatch.

170 full-range INT8/FP64 boundary checks (including ten repeated smoke profiles) and40 representative performance-output checks PASS. Ten kernels use243-244 VGPR,64KB LDS,zero scratch/spills. Physical GPU7,three HIP-event samples,warmup2/repeat5,median. Boundary input clones and guards are checked; representative output verification uses benchmark inputs and guards.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|12160|43.423897|455.768|574.971|0.7927|
|16384|4096|6144|8|12081|15.383708|428.835|636.582|0.6737|
|2048|6144|7168|96|12081|39.323864|440.377|575.891|0.7647|
|2048|4096|8192|128|12161|34.482937|510.171|601.440|0.8482|

All170 boundary and40 representative output checks pass. Batching improves these representative best results over iteration64, but no fixed target is met. The best choices vary by shape; this is a development screen, not a public selector. Do not replace the current full1178 CSV with oracle candidates. No default promotion.

The own-kernel ISA comparison confirms fewer static s_waitcnt instructions as batch grows while native MMAC and LDS-read counts stay equal. These are whole-kernel static counts, including fallback and epilogue, not dynamic cycle percentages. Resource metadata stays243-244VGPR with zero scratch/spills. Raw disassembly stays outside Git; opcode summaries are archived. Next evaluate deferring only a bounded final A batch across stage barriers, to overlap MMAC with the following B prefetch without retaining the whole A tile.
