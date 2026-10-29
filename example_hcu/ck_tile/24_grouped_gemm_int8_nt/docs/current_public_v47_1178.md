# Current default public implementation: all1178 INT8 NT shapes

Physical GPU7,INT8xINT8->INT32,no scale. All rows use the identical public-v47 library SHA256 recorded in the CSV. Reuse560 prior small-M results after verifying the library identity and dimensions; measure the remaining618 through the default public selector. No experiment-oracle winner substitution. Each row identifies its measurement source. All timings use10 warmups,50 repetitions and three HIP-event samples/median.

Every output is checked with inputs in[-8,8],FP32 exact-reference arithmetic and TF32 disabled; validation is chunked to avoid a whole-output reduction memory spike. Guard regions and sampled input immutability are checked. This is not a full-range exhaustive correctness claim;28 full-range boundary checks are separately recorded in iteration47. TOPS counts multiply/add as two operations. Fixed BLAS targets remain the workbook columnK values.

{
  "numeric_pass": 1178,
  "ge1": 832,
  "ge05": 1164,
  "min_R": 0.45631943431758887,
  "median_R": 1.198147838917571,
  "library_sha256": "865755915e5f4b9045d40689311ca531f7980b7531b9ac0b4cf22748a51e9bb5",
  "phase": {
    "decode": {
      "total": 872,
      "ge1": 825,
      "median_R": 1.329005798024912
    },
    "prefill": {
      "total": 306,
      "ge1": 7,
      "median_R": 0.607738874675903
    }
  }
}
