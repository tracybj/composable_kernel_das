# Iteration109:skip empty N384 consumer work while retaining barriers

Hypothesis:the last N384 block can contain unused32-column consumer stripes. Skip their initial/next B reads,LDS A reads andMMAC while retaining every publication/reuse barrier;the existing epilogue already skips their stores. No producer or consumer returns before the final shared synchronization. Other geometry,packed format1 and MGroup4 stay fixed. Added independent Problem/Pipeline/Kernel/Epilogue in wasp_masked_n_v109. The control's four v107 headers match the prior compiled-source hashes.

Build succeeds:both old and masked use182VGPR,zero spill,64KiB LDS;masked uses32 rather than30SGPR.24 full-rangeINT8/FP64 checks PASS,including odd stages,N tails with mostly-empty consumers,padding,ragged/empty groups,guards and immutable inputs.16 representative shapes/32 performance outputs PASS,balanced order,2warmups/5repeats/3sample median.

Some shortK cases improve: M16384/N4096/K1536/G4 goes401.00 to409.48TOPS(+2.11%). However longK gains are small or absent: M16384/N4096/K6144/G8 goes651.37 to650.92;M16384/N4096/K7168/G8 goes672.37 to671.08,remaining below675.51 target. Fully-divisibleN cases also move slightly,so the timing differences cannot be attributed exclusively to skipped tail work without further ablation. The hypothesis does not close the remaining longK gap. Do not promote or launch a full344/1178 run for this candidate.

Continue with true M32 packed geometry to address the two decode gaps,and use the validated unmasked N384/A2 as a separate basis for dimension-selected largeM improvements. A fresh public-API acceptance remains required. Current formal v106 remains1178 numericalPASS,1115 targets;no FP16/BF16 interface or defaults changed.
