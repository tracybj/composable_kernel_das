# Grouped GEMM optimization

Own CK-Tile INT8 NT implementation, experiments and iteration logs.

Code: example_hcu/ck_tile/24_grouped_gemm_int8_nt/

Upstream base: 9acdc34e1f733f3c5dbfc85b52d6273409ec2ce8. Integration changes: integration/ck_int8_nt.patch. Upstream history, binaries and vendor disassembly are not included.

Target: 1178 shapes from BlasLt_ggemm_tuning_baseline.xlsx, int8_NT column K; INT8 x INT8 -> INT32, NT, no scale. Dispatch uses matrix dimensions, group count and supported attributes, never benchmark case IDs.

Iteration 26 public-interface validation: all 560 shapes with M <= 16 pass numerical checks and reach the frozen target in this run (minimum throughput ratio 1.0034, median 1.3412). Eight have less than 3% margin. This does not establish completion of the remaining 618 shapes with M > 16.

See `example_hcu/ck_tile/24_grouped_gemm_int8_nt/docs/iteration_26_public_audit.json` and `iteration_26_public_small_m.csv` for validation evidence. Native direct-to-LDS, ping-pong, LDS mapping and larger-M experiments are recorded separately in the iteration logs and are not yet default production paths.
