# Iteration85:512x128x64 operand-reuse tradeoff without spills

Retain M512,Kstage64,the four producer/eight consumer roles and64KB double-buffer A LDS. Reduce N256 toN128:each consumer wave owns16 output columns,one B fragment row per stage and32x1 MMAC accumulator vectors(128 INT32 values per thread). B loads,consumer stripe,accumulator indexing,epilogue,CTA grid and256x128 packed fallback are adapted together. Both internal mappings,record0/1024,A batches2/4. External INT8 NT/INT32/no-scale math and original group counts are unchanged.

At equal full GEMM work versus public256x256x128,the ideal tile-load model halves B requests but doubles A requests and A LDS reads. Accumulators match the public128-value count. Total requested A+B tile bytes rise about25%,and stage synchronization frequency per output work increases. The cache and latency implications require measurement;less B traffic alone is not a promise of higher throughput.

CPU address/ownership model PASS.216 full-range INT8/FP64 boundary checks and56 representative performance-output/guard checks PASS,including odd/even native stages,padded inputs,ragged lengths,tails and128-wide fallback. Eight kernels report156--173VGPR,64KB LDS,zero private/scratch,zero VGPR/SGPR spills. GPU7,three HIP-event samples,warmup2/repeat5,median.

| M | N | K | G | best candidate | ms | TOPS | public v80 TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|27004|1.912544|143.724|179.977|201.009|0.7150|
|4096|1024|2048|64|27014|4.685439|234.666|332.550|374.064|0.6273|
|2048|1536|2048|8|27014|0.444288|232.010|329.433|414.146|0.5602|
|16384|3072|4096|48|27004|72.898605|271.490|540.364|574.971|0.4722|
|16384|4096|6144|8|27114|23.336666|282.691|574.678|636.582|0.4441|
|2048|6144|7168|96|27104|60.487957|286.293|604.940|575.891|0.4971|
|2048|4096|8192|128|27114|59.674644|294.802|615.706|601.440|0.4902|

Removing spill restores much of the performance lost in iterations83/84,but none of these representative choices beat public v80. The hard M16384,N4096,K6144,G8 case reaches282.7TOPS versus574.7 public v80;the K4096 long-M case reaches271.5 versus540.4. Do not promote the structure. This establishes a negative result for the tested reuse/synchronization arrangement,not all512-row tiles or a hardware limit. Extra A movement/LDS reads and shorter stages are plausible costs,but this run does not individually attribute runtime to each cost.

The three compatible structural trials show that simply increasing M reuse is insufficient here. Next return to the fast256x256x128 geometry and inspect consumer C/CP scheduling under the passing record1024 path. A separate experimental protocol can investigate whether CP's high-half MMAC can carry in registers across a single publication/reuse barrier while its LDS reads finish before release. Preserve the production uniform-C assertion until a compatible runner,odd/even/tail proof,numeric checks,resources and paired performance establish safety and benefit. Public v80 formal975/1178 remains authoritative;203 fixed-target misses require further optimization.
