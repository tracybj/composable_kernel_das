# INT8 NT v158 implementation and acceptance

## Entry points and dispatch

The raw `ck_tile_grouped_int8_nt_launch` provider is registered by
`19_grouped_gemm`. The [cached API](../grouped_int8_nt_cached_api.cpp) owns
weight preparation, plan creation, and stream ordering. When the descriptors
support packing, weight preparation creates the native K64/N16 layout for
reuse. Raw fallbacks continue to use the original NT weights.

[Selection](../instances/grouped_int8_nt_selector.hpp) uses M/N/K/G,
device CU count, group shape uniformity, and descriptor properties such as
output stride and scale mode. Each selected instance launches a grouped
kernel with the expert indexed by the grid. The [scaled dispatcher](../instances/grouped_int8_nt_scaled_dispatch.cpp)
binds BF16 or FP16 epilogues to the chosen integer GEMM instance.

| Path | Main loop and output |
| --- | --- |
| [Cooperative](../kernels/grouped_int8_nt_pipeline.hpp) | Compute waves cooperate in loading the tiles. The standard path stages A and B in LDS and prefetches the next K tile into registers. |
| [Packed cooperative, route 24](../kernels/grouped_int8_nt_packed_pipeline.hpp) | The small-M instance reads A and prepacked B into VGPRs. Its output mapping uses LDS. |
| [K-parallel, route 25](../kernels/grouped_int8_nt_k_parallel.hpp) | A small-M, large-G specialization reads raw NT A and B into VGPRs, splits K among waves, and merges INT32 partials in LDS before conversion. |
| [WASP](../kernels/grouped_int8_nt_wasp_pipeline.hpp) | Producer waves stage logical A in double-buffered LDS. Consumer waves fetch logical B into registers and run MMAC. Logical transpose swaps the operand roles internally and preserves the original output coordinates. |

Routes 26–30 pair WASP instances with the
[paired-store epilogue](../kernels/grouped_int8_nt_scaled_epilogue.hpp)
for supported dense output and scalar activation scale. It applies the scale
to the INT32 accumulator, converts and packs 16-bit values, then uses wave
exchange to assemble 16-byte stores. v158 batches scale loads and uses
packed BF16 round-to-nearest-even with quiet-NaN handling. Other supported
descriptors use the applicable generic epilogue. See the
[scale contract](INT8_SCALE_OUTPUT.md) for exact operation order.

## v158 results

Physical card 7 ran 1191 shapes for both output types, one formal round per
shape. The target is column P, Max ALL, on worksheet `int8_NT`. Its values
combine BLASLt modes and are used as a throughput threshold. The
[frozen data](../benchmarks/data/targets_max_all_1191.csv) records each shape
and target.

| Output | Numeric PASS | Decode at target | Prefill at target | Minimum CK/target |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1191/1191 | 881/881 | 310/310 | 1.002975 (ID 705) |
| FP16 | 1191/1191 | 881/881 | 310/310 | 1.003342 (ID 1087) |

The run had zero failures or skips. The
[consolidated CSV](benchmark/v158/results.csv) contains both outputs,
[manifest](benchmark/v158/manifest.json) records the run identity, and
[completion summary](benchmark/v158/done.json) records acceptance totals.
These results belong to v158 as built before the source-layout cleanup.
The CSV uses LF line endings in Git. Its original SHA-256 was
`4ab0f596a58910331b015c14eae44a099e53c370c8c1de01f56f0cf50c837f14`.

## v159 consolidation and instance slimming

1. **Unused route elimination**:
   - `route 1` (RawM256N256) and `route 2` (RawM256N256Record1024) in both INT32 and scaled dispatches wrapped in `#ifdef CK_INT8_CACHED_TUNING`.
   - INT32 `route 22` and `route 23` in `cached_dispatch.cpp` wrapped in `#ifdef CK_INT8_CACHED_TUNING`.
   - Total production device kernels in `libck_tile_int8_nt_cached.so` reduced from 142 to 134 kernels (-8 kernels).
   - INT32 none-scale kernels: 44 -> 40 kernels.
   - Scaled kernels: 98 -> 94 kernels (47 compute configurations x 2 data types BF16/FP16).

2. **Prefill shape consolidation**:
   - Shape $N=1536, K=2048$ (ID 699, $M=4096, G=8$, formerly the sole `route 1` case) merged into `route 30` (`PackedM128N256` PairedStore).
   - ID 699 performance increased from 450.5 TOPS ($R=1.0687$) to 507.9 TOPS ($R=1.2031$) on BF16 (+12.6%), and from 463.2 TOPS ($R=1.0974$) to 513.8 TOPS ($R=1.2168$) on FP16 (+10.7%).
   - Exact numerical match verified against FP64 reference.

3. **Gate validation**:
   - `selector_policy_test.cpp`: PASS.
   - `test_scaled_api.py`: PASS.
   - `test_cached_api.py`: PASS (65 checks).
   - `test_lifetime.py`: PASS (32 checks).
   - `test_k_parallel_lifetime.py`: PASS (24 checks).
   - Critical boundary shapes tested on physical GPU 7 with 100% pass rate ($R \ge 1.0$).

## v160 Route 6 to Route 30 paired store epilogue optimization

1. **Systemic Route 6 -> Route 30 upgrade**:
   - In `int8_cached_paired_store_route`, enabled `case 6: return 30;` and broadened guard bounds to `n >= 256` and `k >= 512`.
   - Upgrades all 40 shapes formerly on `route 6` (`PackedM128N256` generic epilogue) to `route 30` (`PackedM128N256` paired store epilogue).
   - Zero new kernels added: `route 30` was already compiled in the binary. Device kernel count remains 134.

2. **Benchmark results across all 40 affected shapes**:
   - 40 / 40 shapes (100%) faster on both BF16 and FP16.
   - Mean BF16 speedup: **+15.37%** (up to +23.7%).
   - Mean FP16 speedup: **+18.40%** (up to +29.6%).
   - Minimum R within the group increased from 1.0145 (ID 651) to **1.1049**, doubling safety margin.
   - Bit-exact numerical match verified against FP64 reference across all groups and shapes.

3. **Validation**:
   - `selector_policy_test.cpp`: PASS.
   - `test_scaled_api.py`: PASS.
   - `test_cached_api.py`: PASS (65 checks).
   - `test_lifetime.py`: PASS (32 checks).
   - `test_k_parallel_lifetime.py`: PASS (24 checks).
