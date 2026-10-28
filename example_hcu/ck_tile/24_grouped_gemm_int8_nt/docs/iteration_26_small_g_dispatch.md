# Iteration 26: small-G long-K public dispatch

Add masked 16x32x512/two-wave instance 6505. Selection requires common nonempty/nonragged M<=16, K>=4096 and G<=8. Other small-M requests keep iteration-23 rules; larger-M requests unchanged. All original shape/alignment/arch validation and C ABI structures are preserved.

Hypothesis: the narrow N tile increases output CTA count for small G; BK512 amortizes stage/barrier overhead. Fifteen previous under-target shapes pass full-output screening and exceed fixed spreadsheet throughput with this candidate. This is not proof for the full domain.

Fresh public library build and host checks pass (80 original, 24 INT8). Twenty-one full-range INT8/FP64 boundaries pass, including nondefault streams, padded strides, K16 tails and guards. Public 560-shape verification is launched next: median of 3 samples, warmup10/repeat50. Full 1178-shape target remains incomplete.

## Completed default-entry validation

560/560 numerical PASS, 560/560 reach the fixed spreadsheet target. Minimum R=1.003401316; median R=1.341192160; 552/560 have R>=1.03. Three HIP-event samples, warmup10/repeat50. Full 1178-row CPU selector replay passes; this is not GPU verification of the other 618 rows. Detailed files: iteration_26_public_audit.json, iteration_26_public_small_m.csv and selector_contract_v26_audit.json.
