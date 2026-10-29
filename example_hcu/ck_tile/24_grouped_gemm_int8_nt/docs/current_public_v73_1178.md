# Current default public implementation: all1178 INT8 NT shapes

Physical GPU7,INT8xINT8->INT32,no scale. All rows use the identical public-v73 library SHA256 recorded in the CSV. All1178 records are newly measured through the public default selector; no reused previous-library records. No experiment-oracle winner substitution. Each row identifies its measurement source. All timings use10 warmups,50 repetitions and three HIP-event samples/median.

Every output is checked with inputs in[-8,8],FP32 exact-reference arithmetic and TF32 disabled; validation is chunked to avoid a whole-output reduction memory spike. Guard regions and sampled input immutability are checked. This is not a full-range exhaustive correctness claim;34 full-range public boundary checks are separately recorded in iteration73. TOPS counts multiply/add as two operations. Fixed BLAS targets remain the workbook columnK values.

{
  "numeric_pass": 1178,
  "ge1": 915,
  "ge05": 1178,
  "min_R": 0.7617497991099058,
  "median_R": 1.2049133250337323,
  "library_sha256": "cc5388e0e0d731a9775b8e4b2c4e8576ca62a4a284bc0f2cbdcd7449d086d16b",
  "phase": {
    "decode": {
      "total": 872,
      "ge1": 844,
      "median_R": 1.3274243393254848
    },
    "prefill": {
      "total": 306,
      "ge1": 71,
      "median_R": 0.9398759626992711
    }
  }
}
