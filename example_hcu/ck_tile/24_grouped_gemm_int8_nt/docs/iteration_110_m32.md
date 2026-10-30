# Iteration110:true M32 packed WASP geometry

Added independent Problem/Pipeline/Kernel/Epilogue. One producer publishes32rows of raw A per128K stage using4 vector loads per lane into4KiB;double buffering uses8KiB LDS. Each consumer owns32x32 output with16INT32 accumulators. N64/128/256 uses2/4/8 consumers. Packed B retains approved format1;raw B variants isolate layout from geometry. Input activations are not packed. No FP16/BF16 interface/default changes.

CPU producer/consumer byte-address model PASS.96 full-rangeINT8/FP64 checks PASS,covering odd K-stage counts,padding/wide strides,M/N tails,ragged/empty groups,input immutability and guarded outputs. All8 kernels use62VGPR,zero spill,8KiB LDS. All104 original M32 requirements,904 performance outputs PASS,2warmups/5repeats/3sample median,balanced order. Four non-native N/K requirements retain legacy in the screen. Several highG cases prefer legacy,so packed M32 is not universally selected.

Dimension-only draft:when packed weights are available,K>=3072 andG<=12,useN256/A2 ifN>=4096 andK>=4096,otherwiseN64/A1;keep legacy elsewhere. It promotes11 requirements,reaches all104 targets,and has no same-run regression in this screen. This is calibrated development evidence,not independent acceptance.

M32/N3072/K4096/G4:52.3198 to76.7836TOPS(+46.76%),target52.5863. M32/N4096/K7168/G8:63.7052 to77.9300TOPS(+22.33%),target63.8156. Raw versions largely fail to reproduce packed gains,supporting the value of contiguous prearranged B fragments in this geometry. The exact attribution still includes coupled changes to geometry and staging versus the legacy kernel.

Next add explicit INT8-only routes with the same packed-weight lifetime contract,then validate full public API. Also compare the wideN candidate against each actual current route before formulating a largeM selection. Formal v106 remains1178 numericPASS,1115 targets until fresh acceptance completes;do not substitute104-case draft counts for full1178 results.
