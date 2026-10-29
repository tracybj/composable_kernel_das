# Iteration84: eight-wave cooperative512x256x64 structure

Keep the iteration83 mathematical tile and64KB double-buffer A LDS,but remove dedicated producer waves. All8 waves/512 threads copy64 A rows each(four16-byte raw-to-LDS loads per lane) and compute512 output rows by32 columns. Load initial A/B,wait and publish;each stage prefetches next A and B before current MMAC,then waits for VMEM and synchronizes publication/reuse. Current A/B are not overwritten before their consumption finishes. Static alternating register/slot references avoid dynamic accumulator indexing. Both mappings,record0/1024 and A batches1/2. A separate512-thread packed fallback covers unsupported domains.

CPU address/ownership model PASS;216 full-range INT8/FP64 boundary tests and56 performance-output/guard checks PASS. Odd/even native stages,tails,padding,ragged lengths and fallback included. Timing:GPU7,three HIP-event samples,warmup2/repeat5,median. Public v80 code and formal CSV unchanged.

| M | N | K | G | best candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|26111|4.301407|63.904|201.009|0.3179|
|4096|1024|2048|64|26111|13.827261|79.518|374.064|0.2126|
|2048|1536|2048|8|26111|1.320960|78.034|414.146|0.1884|
|16384|3072|4096|48|26001|224.592010|88.121|574.971|0.1533|
|16384|4096|6144|8|26001|72.542000|90.941|636.582|0.1429|
|2048|6144|7168|96|26111|220.596466|78.502|575.891|0.1363|
|2048|4096|8192|128|26001|192.793396|91.249|601.440|0.1517|

The eight-wave allocation still spills:declared255--256VGPR,private/scratch244--484 bytes per work item,nonzero VGPR spill metadata and one SGPR-spill field in each kernel. This compatible change does not make the256-accumulator structure competitive;do not promote it or lower target values. Changing producer ownership alone is insufficient in this build. The observations do not prove a universal hardware ceiling or isolate every source of time.

Next reduce N tile to128 while retaining M512/K64. Each of8 consumer waves then owns only16 output columns,bringing accumulators back to128 values per thread. At equal full work,this requests about half as much B and twice as much A as the public256x256x128 tile,with twice the A LDS read volume. It trades operand reuse directions rather than promising lower total traffic. Adapt B loads,MMAC/epilogue stripe,grid and128-wide fallback together,then validate resources/math/performance. All1178-target work remains active;public formal count stays975/1178.
