# M256/N384 hypothesis,not yet measured

Keep the same format1 MMAC B fragments and A publication protocol. Increase consumer waves8 to12,while retaining4 producers and64KiB LDS. Each wave retains256x32 output/128INT32 accumulators;CTA output width grows256 to384. Total B byte requests and MMAC work stay fixed for divisibleN,while A publications/CTA count decrease by one third. A 1024-thread workgroup may increase register pressure or limit scheduling;resource metadata and paired timings must decide.

NativeN is a multiple32,with out-of-N consumer epilogue stripes skipped. Extra consumers may issue buffer-bounded reads whose values are unused;the input descriptor bounds memory and output stores are guarded. Non-native M/K/strides retain the full scalar diagnostic path. Full-range math and N-tail guards are required before measuring. No production/default promotion is implied.

Initial C12/A4 IDs39001/39008 failed multi-stage numerics and are quarantined;see iteration_107_wide_n.md. C12/A2 and C8 variants pass60 checks;do not enable unvalidated variants.
