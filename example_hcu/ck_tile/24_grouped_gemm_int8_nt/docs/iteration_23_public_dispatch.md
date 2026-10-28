# Iteration 23: public small-M shape dispatch

Register three packed instances through the existing select/query/run ABI:

- M<=16, common positive groups without ragged/empty flags, K<1024: 16x64x128, two waves.
- Same domain, K>=1024 and G<=16: 16x64x256, two waves.
- Same domain, K>=1024 and G>16: 16x128x256, four waves.

All original dimension/alignment/architecture validation remains in effect. Masked loads/stores support K16 and N4 tails. Ragged/empty requests retain the original fallback. No case ID/phase dispatch. Original enum values and ABI structures preserved; only new instance values appended.

The shallow K/G split selects 546/560 target-meeting candidates on iteration-22 screening data. This is training evidence; independent public validation is still required. Host selector checks: 21 INT8 plus 80 original cases pass. Fresh standalone public library builds. Eighteen full-range INT8/FP64 boundary tests pass with stride/tails, guards, full input comparison, nondefault stream and errors.

Full public 560-shape verification is launched next with 3 samples, warmup10/repeat50, independent full FP32 exact-integer references on [-8,8], guards and input sampling. Final 1178-shape target is still incomplete. Build: build/public_v23; output: results/public_v23_small_m. No claim of full root build or full FP16/BF16 GPU regression.
