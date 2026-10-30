# Iteration113:balanced-resource tiles and shortened A live ranges

Tested M128/N192 with6consumers/2producers atA4 andA2;M256/N320 with10consumers/4producers atprefetchedA2 ornon-prefetchedA4;andM256/N384 with12consumers/4producers atnon-prefetchedA4. N-stripe output guards preserve partialN correctness. All reuse format1,existing publication/reuse barriers,andraw A. Current actualv112 routes are compiled in the same module;all source dependencies are checked before/after build.

255 full-rangeINT8/FP64 checks PASS across12 current routes and5 candidates,including odd stages,padding,N tails,ragged/empty groups,guards andimmutability. NewM128 A4/A2 use134/118VGPR and32KiB LDS. The3 M256 candidates use182VGPR and64KiB LDS. Allzero spill/private bytes.20 representative original shapes,105 timed outputs PASS,including all8 remaining targets.2warmups/5repeats/3sample median androtated order.

None ofthe8 remaining targets improves overthe current path. ForM16384/N2048/K768/G4,the current253.01TOPS remainsbest;newM128/N192 is about6.4%slower. ForM16384/N4096/K7168/G8,current672.22 remainsbest;non-prefetchedA4/N384 is about8.1%slower. M256/N320 variants also regress. An already-passing M128/N3072/K4096/G4 improves213.72 to232.88TOPS,butthis doesnot solve a missing target. No full422/1178 rerun or default promotion is justified.

Lower registers andapparently balanced block sizes didnot predict improvement. Per-SIMD wave placement,copy/compute request schedules and cache traffic need stronger evidence;this experiment doesnot prove anoccupancy explanation or hardware ceiling. Preserve the validated v112 policy. Next explore a differentrole assignment:cooperative A publication bycompute waves with a widerN output tile,usingpacked B andM256 reuse. Prior raw-B cooperativeM128/K64 failure doesnot rule outthis differentgeometry/dataflow combination. Its synchronization andresource costs must be tested from scratch.

Formal report remainsv112:1178 numericalPASS,1170 targets,all872decode pass,and8prefill gaps. ExistingFP16/BF16 source/default/ABI remainsunchanged. See current_cached_v112_1178.csv anditeration_112_remaining.csv.
