# Iteration70: original344 large-M screen (running)

Freeze the passing iteration69 library, unchanged original M/N/K/G from targets_int8_nt.csv. Screen six single-barrier/A-prefetch candidates (batch2/4/8,original/internal-transposed) plus the current default instance as same-library control. No caseID-based production dispatch. Rotate development candidate order by test sequence,three HIP-event samples,2 warmups/5 repetitions,median. Every output is compared using bounded256-row chunks,guards and sampled input immutability. Inputs[-8,8],FP32 exact reference,TF32 disabled. Keep actual group counts; no memory-budget cropping. Performance screening is not public selector acceptance.

Run directory: results/wasp_one_barrier_v69_large_screen. Inspect progress.json,active.json,records.jsonl,run.log,done.json and live process state. Any original-shape failure stops screening; do not count it as passed. Results and dimension-based rule analysis will be committed after completion.
