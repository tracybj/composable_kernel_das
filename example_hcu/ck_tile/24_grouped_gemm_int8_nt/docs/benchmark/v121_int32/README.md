# v121 INT8 × INT8 → INT32 Grouped GEMM

This archive records the gfx936 NT implementation with signed INT8 inputs,
INT32 accumulation and output, and no scales. Expert weights were prepacked
once and reused. GEMM timing excludes weight preparation, which the CSV
reports separately.

v121 consolidated the implementation into the cooperative and parameterized
WASP main loops. The original 1178 shapes kept the same selected instances
as v112.

| Phase | Shapes | Numeric PASS | CK/BLASLt ≥ 1.0 |
| --- | ---: | ---: | ---: |
| Prefill | 306 | 306 | 298 |
| Decode | 872 | 872 | 872 |
| Total | 1178 | 1178 | 1170 |

The frozen BLASLt throughput target is the denominator. The minimum ratio
was 0.981198. Card 7 ran one formal round with 10 warmups and 50 repetitions
per sample, reporting the median of three samples. These results are for
INT32 output without scales.

- [Full results](results.csv), [summary](summary.json), and [final audit](audit.json).
- [Published v121 source snapshot](https://github.com/tracybj/composable_kernel_das/tree/4f2991c1c8829d6b5f248c71adcfe0eacb3f9997/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
- [Historical report and reproduction notes](https://github.com/tracybj/composable_kernel_das/blob/4f2991c1c8829d6b5f248c71adcfe0eacb3f9997/example_hcu/ck_tile/24_grouped_gemm_int8_nt/docs/REFACTOR_TWO_CORE_v121.md).

The run recorded server revision `2a16d588b39d024e7f5646bde152fc0158bc313b`.
The published snapshot contains the matching v121 pipeline and cached-weight
source trees from the server delivery archive `3c25c649`. The final audit
records the byte-identical build after the default-parameter correction.
