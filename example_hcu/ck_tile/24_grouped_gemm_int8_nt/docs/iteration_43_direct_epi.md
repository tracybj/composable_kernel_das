# Iteration43: packed input plus direct vector output

Combine the existing masked packed global-prefetch pipeline with the scalar/wave-transposed no-LDS epilogue. New typed problems and kernel wrappers retain existing arithmetic, capacities and stride semantics. No default rule change.

128 full-range INT8/FP64 combinations pass, with full input clones and output guards. All834 M<=128 target shapes and4196 candidate-output checks pass using independent full-output [-8,8]/FP32 integer reference, guards and256 input samples per operand. Performance is median of3 HIP-event samples,warmup3/repeat10. This is development candidate screening, not public default acceptance.

Best-candidate target coverage830/834: M<=16 560/560 (minimum R1.0180), M32 103/104, M64 90/92, M128 77/78. Four residual rows remain unmodified in the audit and CSV. A narrow-N K512 direct-output candidate reaches all190 long-K (K>=4096) small-M targets; a shape rule can be evaluated independently. Arbitrary per-case fastest choices are not a supported production solution.

See iteration_43_direct_epi_audit.json and iteration_43_direct_epi_best.csv. Raw samples and identity: results/direct_epi_v43_screen. Build recipe: experiments/packed_direct_epi_v43/build_command.json. Further work must address remaining small-G latency cases and the large-M compute/copy pipeline; all1178 completion remains unproven.
