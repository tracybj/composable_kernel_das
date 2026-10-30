# Iterations101/102: broad M128 and M256 policy screening

Use the frozen iteration100 library SHA256 `ac7b3f1f59fea159d0a44aa0fc3cd2dc9aa36c8bd6370ab986594329bf4dd1ab`,physicalGPU7,warmup2/repeat5/3samples median,balanced rotating order. No public/default,FP16/BF16 or compiled kernel source changes. All original group counts/shape dimensions retained. One-group FP32 exact references and bounded256-row comparisons avoid full-G reference/reduction temporary OOM;memory preflight fails rather than reducing G.

Iteration101 covers all422 original M>=128 requirements:78M128+344M>=256. All1523 numerical/performance records PASS. Current-policy same-run control attains345/422;oracle attains366/422. These development controls differ from the earlier formal run's341/422;do not splice their results into formal1178 counts.

All74 native M128 requirements improve with packed M128/N128 instance36402:minimum speedup1.1119,median1.4199 over same-build controls. Four non-native tails retain the original provider. Draft M128 result78/78 reach the frozen target,minimumR=1.450676. Packed N256 batch2 is not appropriate as a general M128 choice.

Iteration102 independently covers all38 M256 requirements with raw-transposed record0/1024 and packedN128 alternatives. All152 records PASS. Current-policy same-run control reaches33/38,oracle reaches38/38. Raw transposition is not a general replacement;some long-K/G configurations regress substantially. The validated packedN128 family provides the needed targeted choices.

## Dimension-only draft

In the supported uniform/nonempty native domain with the required packed weight available:

```cpp
if(M == 128) return packed_m128_n128_a4;
if(M == 256 &&
   ((N == 3072 && K >= 1536) ||
    (K == 2048 && (N <= 1024 || (N <= 1536 && G >= 16)))))
    return packed_m128_n128_a4;
if(M >= 8192 && K == 512) return packed_m128_n256_a2;
if(M >= 8192 && K == 768) return packed_m128_n128_a4;
if(M >= 2048 && K == 2048 && N <= 1024) return packed_m128_n256_a2;
return current_policy;
```

The draft uses dimensions,not requirement IDs. Unsupported/non-native/ragged/empty cases keep their existing fallback. It reaches365/422 in the same calibration screen,minimumR=0.958985,with zero>3%regressions versus controls. This is a calibrated candidate policy,not independently accepted public behavior. Table/caseID values identify benchmark records only and are not embedded in kernels or selectors.

## Next integration

Add the two packed variants to the independent INT8 cached-weight module;keep all original FP16/BF16 ABI structs,entrypoints,layouts and default paths untouched. Extend one-time packing to the newly beneficial K512/K768 families. Ensure plan argument binding uses packed B for every new packed route,not only the old route5. Preserve raw weights and lifetime/cross-stream contracts. Rebuild to a new directory,fresh source/library hashes,new API boundary and shared-weight tests,then a new complete1178 public-interface acceptance. No all-target completion claim:formal iteration99 remains1093/1178 with85 gaps until that acceptance is done.
