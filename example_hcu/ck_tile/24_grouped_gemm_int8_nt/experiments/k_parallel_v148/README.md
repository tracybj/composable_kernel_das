# K-parallel INT8 NT prototype

Experimental only;not connected to the production selector or public ABI.
Problem/Pipeline/Kernel/Epilogue use CK-Tile launch and its existing INT8 MMAC.
Each lane loads 16 consecutive K bytes for each operand. The MMAC's logical
row/column coordinates encode K partitions;only diagonal accumulators are kept.
Wave sums are combined through a small shared INT32 partial array and one CTA
barrier. Scale and the existing BF16/FP16 rounding operation occur after integer
reduction. No global scratch,atomics,dynamic activation packing or B prepack.

Domain:uniform M1 or M2,positive common N/K,K and input row strides divisible
by16. N may have a partial output tile;K may have a partial CTA span. This
test adapter borrows device pointers and does not implement the production
API's full validation/lifetime contract. The original API remains unchanged.

Variants 0/1/2/3 are respectively waves/columns 8/8,8/4,4/8,4/4.
These are dimension-independent tuning variants,not caseID dispatch.

Build using `python3 build.py --build <isolated-build-dir>`.
Run `python3 test_mapping.py` before GPU gates.
With physical GPU7 selected and CK_INT8_CACHED_BENCH_LIB pointing to the frozen
v145 benchmark library,run `python3 test_gpu.py <result-dir> <libk_parallel.so>`.
Append `formal` for fixed variant1 versus v145,10warm/50repeat/3samples.
The10 target IDs identify benchmark fixtures only. Timing covers the complete
grouped call;descriptor preparation is outside it for both implementations.

Full1191 production acceptance is required after any later selector integration.
