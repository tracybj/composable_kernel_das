# Iteration41: frozen BLASLt replay and real launch geometry

Reuse the frozen tuning package and algorithm6950 (GroupedGemm), INT8 NT -> INT32, null scales, M2048,N4096,K8192,G128 on physical GPU7. No tuning, target change or baseline overwrite. All output elements, output guards and regenerated full input buffers pass in both profiler and normal runs.

Separate no-profiler timing:3 samples warmup5/repeat20 = [29.152591705322266, 29.27128028869629, 29.13718032836914]; median29.152592ms, 603.452TOPS. The frozen workbook target is601.440TOPS, so the target is reproducible on the working card.

Actual PMC launch:768 threads per block,65536 LDS bytes,no scratch,256 profiler architectural VGPRs. ELF metadata for the selected GroupedGemm and large Matmul kernels declares max workgroup768,LDS65536,VGPR255,zero spill counts. Thus the gap cannot be attributed simply to our using more LDS/registers. Metadata maximum and actual launch are recorded separately.

Raw reported SQ_LDS_BANK_CONFLICT is0 for this BLAS dispatch versus nonzero in the earlier CK capture. Counters are not yet interpreted as a conflict percentage or a causal bottleneck; output-layout and matched CK counter experiments are needed. Raw CSV remains in results/profile_v41_blas. The profiler's event time is heavily perturbed and is excluded from performance ranking. Self-generated code/metadata summaries are preserved; raw vendor ISA is not published.
