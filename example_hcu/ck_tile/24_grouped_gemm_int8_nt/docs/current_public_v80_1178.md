# Current default public implementation: all1178 INT8 NT shapes

Physical GPU7,INT8xINT8->INT32,no scale. All rows use the identical public-v80 library SHA256 recorded in the CSV. All1178 records are newly measured through the public default selector; no reused previous-library records. No experiment-oracle winner substitution. Each row identifies its measurement source. All timings use10 warmups,50 repetitions and three HIP-event samples/median.

Every output is checked with inputs in[-8,8],FP32 exact-reference arithmetic and TF32 disabled; validation is chunked to avoid a whole-output reduction memory spike. Guard regions and sampled input immutability are checked. This is not a full-range exhaustive correctness claim;44 full-range public boundary checks are separately recorded in iteration80. TOPS counts multiply/add as two operations. Fixed BLAS targets remain the workbook columnK values.

{
  "numeric_pass": 1178,
  "ge1": 975,
  "ge05": 1178,
  "min_R": 0.7954511469116752,
  "median_R": 1.211549239005743,
  "library_sha256": "10f21a6d71ed11497c9a3c72b621d8e7a717cb3c7c7294fdc009f2576e1ee04e",
  "phase": {
    "decode": {
      "total": 872,
      "ge1": 846,
      "median_R": 1.329779528568767
    },
    "prefill": {
      "total": 306,
      "ge1": 129,
      "median_R": 0.9915499524803606
    }
  },
  "comparison_to_v73": {
    "previous_ge1": 915,
    "newly_ge1": 62,
    "lost_ge1": 2,
    "new_policy_shapes": 168,
    "new_policy_ge1": 92
  },
  "by_M": {
    "1": {
      "total": 112,
      "ge1": 112,
      "median_R": 1.36935341541342,
      "min_R": 1.0361658763640191
    },
    "2": {
      "total": 112,
      "ge1": 112,
      "median_R": 1.3677251304888212,
      "min_R": 1.034346934975136
    },
    "4": {
      "total": 112,
      "ge1": 112,
      "median_R": 1.3727935033615677,
      "min_R": 1.0367730290958077
    },
    "8": {
      "total": 112,
      "ge1": 112,
      "median_R": 1.3682783413311945,
      "min_R": 1.0359230721795651
    },
    "16": {
      "total": 112,
      "ge1": 111,
      "median_R": 1.3246991879532222,
      "min_R": 0.9095332555794282
    },
    "32": {
      "total": 104,
      "ge1": 102,
      "median_R": 1.3635964388440929,
      "min_R": 0.9751164835920785
    },
    "64": {
      "total": 92,
      "ge1": 90,
      "median_R": 1.418472206556218,
      "min_R": 0.9441828365791097
    },
    "128": {
      "total": 78,
      "ge1": 69,
      "median_R": 1.2013337617988888,
      "min_R": 0.8641217615658141
    },
    "256": {
      "total": 38,
      "ge1": 26,
      "median_R": 1.0913950300108366,
      "min_R": 0.9580386643474805
    },
    "2048": {
      "total": 108,
      "ge1": 66,
      "median_R": 1.0231299568943282,
      "min_R": 0.7954511469116752
    },
    "4096": {
      "total": 88,
      "ge1": 49,
      "median_R": 1.0242692539004492,
      "min_R": 0.8890193284509845
    },
    "8192": {
      "total": 72,
      "ge1": 14,
      "median_R": 0.9709951193604038,
      "min_R": 0.9217822527123688
    },
    "16384": {
      "total": 38,
      "ge1": 0,
      "median_R": 0.9440490509612411,
      "min_R": 0.8953672531051114
    }
  },
  "remaining": {
    "total": 203,
    "small_M_lt256": 14,
    "large_M_ge256": 189
  }
}
