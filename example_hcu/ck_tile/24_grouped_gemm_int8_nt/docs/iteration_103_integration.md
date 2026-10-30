# Iteration103: integrate the two validated M128 packed routes

Add routes6(M128/N256/A-batch2)and7(M128/N128/A-batch4)to the independent signed INT8 NT→INT32/no-scale API. The dimension-only draft from iterations101/102 selects these routes;no benchmarkID,target table,phase or workload list is consulted at runtime. Other paths retain the current policy. Uniform nativeM128 becomes eligible;ragged/empty and other unsupported native domains keep the legacy provider.

One-time packing now includes K512/K768 families. The same physical format1 is reused. Plan binding explicitly chooses owned packedB for routes5/6/7;all other routes borrow original immutable NT weights. Raw source buffers,A/C and used streams must remain valid for the documented plan/weight lifetime. Opaque handles must not be mixed between different loaded library versions.

New files are within `cached_weights/m128/`;all existing FP16/BF16 ABI structs,entrypoints,layouts and default source files are unchanged. New module still exports exactly eight new C APIs and no old grouped-GEMM C entrypoints. Mixed C header ABI checks and originalINT8 host45 checks pass;all nine old public-context source hashes remain unchanged. Same-process loading keeps all four old entry addresses unchanged. This is interface/CPU-loader evidence,not a new FP16/BF16 GPU performance claim.

Build succeeds in `build/cached_v103/`,separate from frozenv97. SHA256 `985b3e3f3f8b337581862457d870dbbabf9ece704482cac82400559b88eb5f54`. Test bindings resolve the actual loaded API symbol via dladdr and check its owner path,so an older same-SONAME library cannot silently satisfy the benchmark adapter.

47 full-range INT8/FP64 boundary cases PASS,including complete route coverage0/1/3/4/5/6/7,padded/raw/wide descriptors,packed bytes,guarded outputs,immutable inputs and nondefault streams. Initial46-case outputs were all correct,but the route-coverage gate failed because new selection consumed the oldraw route3 test profile. Added a dedicated raw route3 profile and reran47;the first log is preserved. Sixteen shared-weight,explicit same-address refresh,multiple-stream and queued-work destruction checks PASS across M128/256/512/2048. CPU1178 input/CSV/resume tests pass.

A new full1178 public-interface acceptance is still required. Do not replace formal iteration99 counts with development-screen counts or claim all-target completion. Oldformal1093/1178 remains the accepted performance report until the new run is audited. Preparation costs and persistent bytes remain separate from steady-state GEMM time.

## Completed public acceptance

All1178 numerical checks PASS;1117/1178 targets met,61 still unmet. MinimumR=0.944792. See iteration_103_cached_api_1178.md and current_cached_v103_1178.csv. The above pending statement records the earlier integration checkpoint.
