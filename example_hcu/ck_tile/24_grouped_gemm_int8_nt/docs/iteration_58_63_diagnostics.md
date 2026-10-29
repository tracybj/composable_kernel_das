# Iterations58-63: INT8 WASP correctness isolation

Scope: experimental adaptation of the current BF16 WASP dataflow to INT8 NT / INT32, no scale, physical GPU7. Public selector/library unchanged. No performance acceptance from these diagnostics.

58: direct INT8 carrier adaptation compiled, but full boundary suite did not complete; not promoted. 59: native int32x2 fragments plus compile-time uniform consumer persona. Captured recheck fails at M=N=K256,G1,candidate10001,53568 mismatches; input clones and output guards pass. The initial numerical failure prevents its performance run.

60: standalone thread identity, A direct-to-LDS reconstruction, B fragment export, and raw GEMM scalar-output diagnostic. The initial B export used dynamic fragment indices and produced invalid diagnostic values. It is superseded by61; do not use60 B export to diagnose the production load path. Diagnostic compilation requires readfirstlane wave IDs for scalar m0 operands.

61: static B indices and explicit signed-byte extraction. Thread identity, A reconstruction and B reconstruction all match exactly, for contiguous and padded inputs. Raw GEMM remains incorrect, with errors concentrated in late M tiles. This rules out those isolated load mappings for these two small inputs, not all pipeline scheduling defects.

62: only change from61 is volatile ordered MMAC assembly with a memory compiler barrier. Both isolated input paths still pass; raw GEMM still fails in the final M tile (4096 mismatches in each input profile). Ordering alone does not fix the failure.

63: retain62 ordered MMAC; in the uniform-C consumer read and compute one M fragment at a time, replacing16 simultaneously live A fragments with one. The same eight diagnostics now pass, including full256x256x256 INT8/FP64 exact output comparison for both strides, guard checks and full input clones. This establishes a working controlled configuration, not a proven compiler bug or complete operator correctness. More per-fragment waits may hurt throughput. Next build64 full kernel, four original/transposed mapping and CTA-order configurations; run its complete boundary suite before performance.

All full-range diagnostics use[-128,127] and independent FP64 reference where applicable. Exact source/library hashes and individual outcomes are in iteration_58_63_diagnostics.json.64 is a separate experiment and is not covered by this commit.
