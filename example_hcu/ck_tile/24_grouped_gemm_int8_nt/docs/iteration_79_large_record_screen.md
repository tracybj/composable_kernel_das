# Iteration79: completed all344 original large-M record-stride screen

All344 shapes and1720 candidate outputs PASS. Original M/N/K/G and frozen workbook targets preserved; five candidates per shape: both mappings with record0/1024,and current public-v73 default as same-library control. Three HIP-event samples,warmup2/repeat5,median,physical GPU7. Every output checked in256-row chunks,guards and sampled input immutability. Inputs[-8,8],TF32-disabled exact FP32 reference; full-range validation separately in iteration77.

Raw screening contains an unstable ID343 control:9.698,27.617,5.290ms versus prior public2.607ms. Independent recheck of all five candidates passes and returns public control2.608ms. All original1720 records remain intact; the recheck is separately recorded. The raw minimumR below50% is from this timing anomaly and does not replace current_public_v73_1178.csv. No root cause for the transient timing noise is established.

The development dimension rule keeps power-of-two K on the existing default; non-power-of-two K uses record1024-transposed when M<=256 or K<1024,otherwise record1024-original. This rule uses input dimensions only and is evaluated on the same development data; it is not a held-out or public API acceptance result.

| M | Shapes | Raw default reaches target | Oracle reaches target | Draft rule reaches target |
|---:|---:|---:|---:|---:|
|256|38|26|29|28|
|2048|108|37|67|66|
|4096|88|29|52|48|
|8192|72|3|16|13|
|16384|38|0|1|1|

```json
{
  "scope": "All344 original M>=256 shapes,5 explicit candidates,development screening only; current public formal CSV remains v73",
  "numeric_shapes": 344,
  "numeric_records": 1720,
  "library_sha256": "5a5bf4c9e218ea61ca16e6773ae5856bb62b061cfaccaa05ffc65d0acb48ce16",
  "raw_control": {
    "total": 344,
    "ge1": 95,
    "min_R": 0.2967098624867015,
    "median_R": 0.9480424888675709
  },
  "raw_oracle": {
    "total": 344,
    "ge1": 165,
    "min_R": 0.8951966943180666,
    "median_R": 0.9965444576629547
  },
  "dimension_rule_development_only": {
    "total": 344,
    "ge1": 156,
    "min_R": 0.2967098624867015,
    "median_R": 0.9938498117828563
  },
  "dimension_rule": "For non-power-of-two K choose record1024-transposed if M<=256 or K<1024,otherwise record1024-original; power-of-two K retains public default",
  "dimension_rule_regressions_gt3pct": 0,
  "raw_records_sha256": "3a8c748abcd6039ac341b7f89965f649230a052fa885d62a1380537cc7ada027"
}
```

The draft has no >3% regression against same-run control in these344 development shapes. Next integrate the address decomposition as an optional typed pipeline policy,retain original public IDs and fallback,add new dimension-selected IDs and test the real public API across the full1178 domain. All1178 fixed-target completion remains unproven.
