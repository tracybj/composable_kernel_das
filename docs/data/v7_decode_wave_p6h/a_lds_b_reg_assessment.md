# A-LDS/B-register decision record

This design was not previously tested against the A-register/B-LDS path. The old choice was a hypothesis based on the NN operand layout, not an experimentally established winner.

For a 16x128 output tile and one K32 stage, a single shared A panel is 1024 bytes and B is 8192 bytes. If four N-waves instead own private A copies, total A storage becomes 4096 bytes. At M64 those private copies become16384 bytes, larger than the8192-byte B panel. Comparisons must account for wave ownership and equivalent buffered K depth, not just nominal stage count.

A is contiguous in K in NN. The native BF16 MMAC input contains four K values per lane for a fixed M/N coordinate. B is contiguous in N; direct B-register loading can use four scalar K loads per N16 fragment, or coalesced vector loads followed by register redistribution. This is a cost tradeoff, not proof that B must reside in LDS. The repository has a BlockGemmASmemBRegCRegV1 building block, but it is not a ready-made gfx936 decode pipeline or a performance result.

Current first A-LDS/B-register candidate: wave-private A panels, K32, two-stage buffering, four N-waves with complete M16/32/64 ownership. B uses scalar raw-buffer loads with LLVM-visible dependencies. It has no CTA barriers and no cross-wave reduction. This isolates a feasible private-A implementation; it does not test the separate shared-A/barrier scheduling alternative.

The first opaque-assembly B loader was rejected before representative performance: pressure-case correctness failed and ISA showed v_mov of pending VMEM data before a wait. The revised loader uses recognized LLVM buffer intrinsics. These changes may add conservative automatic waits; its timings cannot establish an upper bound on all A-LDS designs. All raw failed evidence is retained.

The comparison scope includes all15 added1179..1193 shapes as active/fallback controls, full output and allocation guards before timing, GPU4 serial, same-instance checks, three rounds, no full1193 replay. Production enable remains off.
