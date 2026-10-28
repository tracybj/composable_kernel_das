# Iteration 36: first native D2L hardware-counter capture

Installed DTK hipprof --pmc --pmc-type3, gfx936 physical GPU7. One existing candidate7204 (256x256x64, eight compute waves, M tile group4) on M2048,N4096,K8192,G128. Library identity is verified against iteration30; source/math is unchanged. Full output reference after profiling passes. Profiling timings are not baseline timings.

Raw evidence in iteration_36_pmc.json; source command in results/profile_v36/manifest.json; raw CSV in results/profile_v36/d2l.csv. Observed LDS65536 bytes, no scratch, profiler arch_vgpr204 (compiler metadata201, allocation/reporting granularity differs). TCC hit/(hit+miss) counter share=0.788292. SQ_WAIT_INST_LDS and SQ_LDS_BANK_CONFLICT are nonzero. These aggregated counters do not by themselves establish the bottleneck or an exact conflict/wait percentage.

Self-generated ISA shows that the generic K loop retains offset calculations, row/K predicates and native/fallback branches. Next compare an explicitly aligned path, then repeat the same counter capture to distinguish instruction overhead, LDS behavior and cache effects. Keep the generic path for tails and large-stride fallback.

Artifact handling correction: objcopy was initially used without an explicit output ELF and rewrote the outer ELF of the iteration30/32 experiment libraries after their tests. Both were rebuilt with the identical commands and recovered their exact original test-time SHA256 (v30:9e7481da7355c2808527d868998d8d41a799fc34f4bf8c0dfad44052fa4b0f32; v32:a7a9383e815de8453cb525a85fdb5c9fefae48627554ec18cc4553a7295ca85d). Original test records were retained; extraction now always writes a separate ELF. Production library was not touched.
