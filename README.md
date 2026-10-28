# Grouped GEMM optimization

Own CK-Tile INT8 NT implementation, experiments and iteration logs.

Code: example_hcu/ck_tile/24_grouped_gemm_int8_nt/

Upstream base: 9acdc34e1f733f3c5dbfc85b52d6273409ec2ce8. Integration changes: integration/ck_int8_nt.patch. Server source checkpoint: f353f5d. Upstream history, binaries and raw benchmark artifacts are not included.

Target: 1178 shapes from BlasLt_ggemm_tuning_baseline.xlsx, int8_NT column K. Current iteration 19 improves five development samples but does not yet meet all targets. See docs/iteration_19_small_m.md and docs/targets_manifest.json under the example.
