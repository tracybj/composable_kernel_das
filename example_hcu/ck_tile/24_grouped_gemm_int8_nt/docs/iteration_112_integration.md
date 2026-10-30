# Iteration112:integrate M32 packed and device-aware M256/N384 selection

Add INT8-only routes10(M32/N256/A2),11(M32/N64/A1),12(M256/N384/A2). M32 is eligible only for packed weights,K>=3072,G<=12;N>=4096 andK>=4096 uses10,otherwise11. Other M32 cases retain legacy. NewM32 headers are undercached_weights/m32 with a validA2 default. Wide headers use a distinct private namespace underwide_n;known spillingC12/A4 and the unhelpful109 mask are not selected or instantiated.

The old selector is retained as the base. Only base routes1/5 withM>=2048,K>=1024 may use12. Padceil(M/256)toMGroup4;estimate old/wideCTA counts usingceil(N/256)andceil(N/384),multiplyG,then round up bydevice CU count. Select12 forwave ratio>=1.47,or baseline>=24waves andratio>=1.44. CU count is saved from the device properties already queried atweight creation,adding no new device-query call. Thresholds are measured calibration,not universal limits. Runtime does not access requirement IDs,phase,target tables or timing records.

All three routes explicitly bind the same owned packed format1. Weight ownership,ready event,stream dependencies,source immutability and plan destruction semantics stay the same. One prepared expert-weight set can serve differentM plans. All opaque handles remain library-instance specific. No FP16/BF16 descriptors,entrypoints,defaults or existing source files are modified.

Build passes with exactly8 INT8 C API entries andzero old ABI owners. All9 old public-context hashes are unchanged;mixedC ABI/tag/offset checks andoriginalINT8 host45 tests pass. Same-process loading preserves all4 old C entrypoint addresses. This is ABI/loader evidence,not a new FP16/BF16 GPU benchmark. The3 integrated kernels match expected resources:M32 uses62VGPR/8KiB LDS,wide uses182VGPR/64KiB LDS;allzero spill/private bytes.

65 full-rangeINT8/FP64 checks PASS,covering every selected route0/1/3/4/5/6/7/8/9/10/11/12,padding/wide raw strides,odd stages,ragged/empty groups,packed bytes,immutable inputs,guarded outputs andnondefault streams.32 lifetime checks PASS acrossshared weight plans,two generations ofexplicit same-address refresh,two streams,and destruction afterqueued work,including newroutes10/11/12.11 CPU selector checks verifydimension-only choice,device-CU dependence andunknown-device fallback. CPU1178 contract/CSV/resume-corruption checks PASS.

Fresh full1178 public-interface acceptance is still required. DevelopmentM32 policy reaches104/104,andlargeM policy336/344,butthese are calibrated screens,not delivery counts. Previous formalv106 remains1178 numericalPASS/1115 targets until the newrun is complete and audited. Pack preparation time/storage remain separate underthe approved static-weight reuse contract. All-target completion remains unproven.

## Completed public acceptance

All1178 numerical checks PASS;1170/1178 targets met,8 still unmet. MinimumR=0.982297. See iteration_112_cached_api_1178.md and current_cached_v112_1178.csv. The above pending statement records the earlier integration checkpoint.
