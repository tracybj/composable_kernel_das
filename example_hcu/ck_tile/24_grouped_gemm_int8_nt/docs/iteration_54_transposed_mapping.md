# Iteration54: equivalent internal transpose and contiguous scalar output

External contract remains A[M,K],B[N,K],C[M,N], INT8 NT to INT32,no scale. Construct an immutable CK UniversalGemmKernelArgs view for internal D[N,M]=B*A^T, retaining the external E pointer/row stride. Weights use the A-only LDS double buffer; activations are read directly by consumer waves. Map fragment coordinates to C[internal_col,internal_row], allowing contiguous scalar stores without a register/LDS output transpose. No tensor copies or changed effective work. Launch grouping uses the internal M/N tile counts, including non-square/tail cases.

66 full-range INT8/FP64 checks (60 boundary combinations plus six repeated smoke checks),32 performance-output records PASS. Six variants declare243 VGPRs,zero spills and zero private bytes. Timing:3 HIP-event samples,warmup2/repeat5,median,physical GPU7.

The mapping is correct but gains are shape-dependent and do not close the frozen target gap. New candidates remain experimental; public small-M default is unchanged. Next test whether hoisted raw-buffer copy addresses and bulk m0 updates reduce the producer-side address overhead.

| M | N | K | G | candidate | ms | TOPS | R to frozen target |
|---:|---:|---:|---:|---:|---:|---:|---:|
|2048|6144|7168|96|9003|36.359318|476.283|0.827035|
|2048|6144|7168|96|9301|38.081436|454.744|0.789635|
|2048|6144|7168|96|9401|36.790970|470.695|0.817332|
|2048|6144|7168|96|9402|36.746170|471.268|0.818329|
|2048|6144|7168|96|9403|37.285820|464.448|0.806485|
|2048|6144|7168|96|9404|38.172699|453.657|0.787747|
|2048|6144|7168|96|9405|36.780693|470.826|0.817560|
|2048|6144|7168|96|9406|47.512634|364.478|0.632894|
|2048|4096|8192|128|9003|36.597721|480.691|0.799233|
|2048|4096|8192|128|9301|36.416924|483.077|0.803201|
|2048|4096|8192|128|9401|36.129692|486.918|0.809587|
|2048|4096|8192|128|9402|36.153275|486.600|0.809059|
|2048|4096|8192|128|9403|36.315609|484.425|0.805442|
|2048|4096|8192|128|9404|37.892952|464.260|0.771914|
|2048|4096|8192|128|9405|36.025307|488.329|0.811933|
|2048|4096|8192|128|9406|47.650200|369.194|0.613851|
|16384|3072|4096|48|9003|46.692886|423.859|0.737184|
|16384|3072|4096|48|9301|47.569908|416.045|0.723593|
|16384|3072|4096|48|9401|49.291512|401.514|0.698320|
|16384|3072|4096|48|9402|49.162132|402.570|0.700158|
|16384|3072|4096|48|9403|59.686676|331.585|0.576699|
|16384|3072|4096|48|9404|49.531410|399.569|0.694938|
|16384|3072|4096|48|9405|49.172596|402.485|0.700009|
|16384|3072|4096|48|9406|56.298195|351.543|0.611410|
|16384|4096|6144|8|9003|14.490941|455.255|0.715155|
|16384|4096|6144|8|9301|14.864477|443.814|0.697184|
|16384|4096|6144|8|9401|15.314301|430.778|0.676705|
|16384|4096|6144|8|9402|15.320093|430.616|0.676450|
|16384|4096|6144|8|9403|14.522878|454.254|0.713582|
|16384|4096|6144|8|9404|15.502172|425.558|0.668504|
|16384|4096|6144|8|9405|15.328669|430.375|0.676071|
|16384|4096|6144|8|9406|18.215643|362.165|0.568921|

Library SHA256: c277c0dbdfeedea34a56b1c0927a165eb70a1277ca0060381880370b48138730

All1178 target acceptance remains incomplete.
