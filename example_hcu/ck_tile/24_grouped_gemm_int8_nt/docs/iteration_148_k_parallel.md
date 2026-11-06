# v148:contiguous-K MMAC diagonal reduction for M1/M2

The actual scaled BLAS winners11845/11805 motivated a new computation mapping,
not another direct-load substitution in a conventional BM16 tile. Each lane
loads16 consecutive physical K bytes from raw NT A/B. Logical MMAC rows and
columns encode K partitions;diagonal accumulators alone form the requested
dot product. Wave reduction and one CTA barrier combine INT32 partial sums,
then the existing FP32 factor/RNE epilogue writes INT32/BF16/FP16 as selected.
Independent CK-Tile Problem/Pipeline/Kernel/Epilogue,not copied vendor ISA.

Domain is uniform M1/M2,K and input strides divisible by16,common positive N/K.
Partial N and partial CTA K spans are masked. No input LDS,packing,global scratch
or atomic reduction. Four experimental variants compare4/8 waves with4/8 output
columns. The test adapter is isolated from the maintained cached ABI/selector;
it does not yet implement the production API's full lifetime/domain contract.

CPU symbolic coverage:49472 K contributions,each counted exactly once.
GPU full-range INT8/FP64 exact-dot gates:140PASS across all four candidates,
INT32/BF16/FP16,NULL/per-row/per-channel/special scales,M1/M2,K16/640/7168/8192/8208,
N tails,padded strides,output guards and unchanged A/B. NaN masks are compared
explicitly. Existing scaled conversion preserves the FP32 factor boundary.
All24 generated kernels have zero private/spill;VGPR39..107,LDS64..512bytes.

First compile failed on a CK constant used as an ext-vector index;using its
integral constexpr value fixed it. The first timing adapter attempt passed all
140 gates but was rejected before baseline launch because group output bases
were not16-byte aligned. Fixing padded group allocation resolved it. Original
logs remain under build/max_all_v148_k_parallel and results/max_all_v148_k_parallel.

Screen:all10 remaining decode shapes×two outputs×five implementations=100
records,all complete outputs PASS. The same variant1(8waves,4columns) wins
consistently. Fixed variant1 was then independently compared with frozen v145
using10warm/50repeat/3samples,rotating backend order on the same buffers and
stream.40 records cover10 shapes×two outputs×two implementations;all PASS.
All20 new output configurations exceed frozen P targets. R min=1.01175754;
same-round speedup min/max=1.04437327/1.23455019. These are one formal timing
round with three event samples,not three independent rounds. No profiler.

Library SHA:f23898da5f8255f965a0aa6dc8f8a4163ec4b20b3c92934b581c7f431d71b590.
Raw GPU results/resources/commands are retained under build/max_all_v148_k_parallel,
results/max_all_v148_k_parallel_retry and results/max_all_v148_k_formal in ck_yyc.
Archived CSV:max_all_v148_k_formal.csv;summary:max_all_v148_summary.json.

Production remains v145;this does not update1191 acceptance counts or splice
local winners into the full CSV. Next integrate a dimension-driven supported
domain into the cached API,verify ABI/lifetime/scale boundaries and wider M1/M2
neighbors,then run independent full1191 acceptance. P1 scaled prefill output
organization and P2 packed N64 compact cooperation remain planned,not tested.
