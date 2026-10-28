# Grouped GEMM optimization

Own CK-Tile INT8 NT implementation, experiments and iteration logs.

Code: example_hcu/ck_tile/24_grouped_gemm_int8_nt/

Upstream base: 9acdc34e1f733f3c5dbfc85b52d6273409ec2ce8. Integration changes: integration/ck_int8_nt.patch. Upstream history, binaries and vendor disassembly are not included.

Target: 1178 shapes from BlasLt_ggemm_tuning_baseline.xlsx, int8_NT column K; INT8 x INT8 -> INT32, NT, no scale. Dispatch uses matrix dimensions, group count and supported attributes, never benchmark case IDs.

Iteration 26 public-interface validation: all 560 shapes with M <= 16 pass numerical checks and reach the frozen target in this run (minimum throughput ratio 1.0034, median 1.3412). Eight have less than 3% margin. This does not establish completion of the remaining 618 shapes with M > 16.

See `example_hcu/ck_tile/24_grouped_gemm_int8_nt/docs/iteration_26_public_audit.json` and `iteration_26_public_small_m.csv` for validation evidence. Native direct-to-LDS, ping-pong, LDS mapping and larger-M experiments are recorded separately in the iteration logs and are not yet default production paths.

Iteration 37 adds five dimension-based packed deep-K instances for M32/64. All 196 public-interface numerical checks pass; 192 reach the frozen target (minimum ratio 0.9442, median 1.3858). The four misses are retained in `docs/iteration_37_public_medium_m.csv` under the example. The registry now has 23 INT8 instances; original FP selector tests and 28 full-range INT8 boundary tests pass.

M128 iteration 40 screening covers all 78 shapes: every output check passes and the fastest available candidate reaches the target on 77 shapes. These are explicit development candidates, not default-selector acceptance. All 1178 target shapes have not yet been completed.

Current public default (iteration 47): 24 INT8 instances, 44 registry entries including FP. A reusable no-LDS vector epilogue is selected for uniform M<=16,K>=4096 at every G; all other dispatch rules are preserved. All560 small-M public numerical checks pass and all560 reach the frozen target in this run (minimum ratio1.0073, median1.3667;557 have at least3% margin). Host80+33,1178 CPU shape replay/relabel/reverse checks and28 full-range INT8/FP64 boundary profiles pass. Evidence: `docs/iteration_47_public_audit.json` and `docs/iteration_47_public_small_m.csv` under the example. This run does not accept the remaining618 shapes.

Iteration48 separates producer and consumer loop/register lifetimes for larger M. All54 checks (48 boundary combinations plus six repeated smoke checks) and36 performance-output records pass; metadata shows233-234 VGPRs without spills. The four representatives reach about425-482 TOPS, below their frozen targets. This remains experimental and is not the public default. See `docs/iteration_48_split_roles.md` under the example.
