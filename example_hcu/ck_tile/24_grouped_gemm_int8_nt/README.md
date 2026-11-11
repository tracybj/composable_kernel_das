# INT8 NT grouped GEMM

Each expert computes `C[M_e,N] = A[M_e,K] × B[N,K]ᵀ` from signed INT8 inputs.
The raw provider and the cached-weight v1 API write INT32. The cached-weight
v2 API also supports fused FP32 scales with BF16 or FP16 output. Expert weights
can be prepared once and reused across calls.

The v158 release on card 7 passed numerical validation and reached the frozen
BLASLt workbook target on all 1191 shapes for both scaled outputs. See the
[implementation and acceptance report](docs/CURRENT_INT8_NT_STATUS_AND_FLOW.md),
[scale contract](docs/INT8_SCALE_OUTPUT.md), and
[full benchmark CSV](docs/benchmark/v158/results.csv).

## Source layout

| Location | Responsibility |
| --- | --- |
| [instances.cpp](instances.cpp) | Raw INT8 provider entry used by `19_grouped_gemm`. |
| [instances/](instances/) | Dispatchers, selectors, routes, and tuning selectors for raw, cached, and scaled routes. |
| [kernels/](kernels/) | INT8 NT kernels, epilogues, WASP/core kernels, K-parallel, and sub-tile problems. |
| [pipeline/](pipeline/) | Shared cooperative and WASP main loops. |
| [grouped_int8_nt_cached_api.cpp](grouped_int8_nt_cached_api.cpp) | Cached weights lifecycle & plan management API implementation. |
| [include/](include/) | Public & internal headers for cached weights, scaling, and K-parallel args. |
| [benchmarks/](benchmarks/) | Reproduction scripts, benchmark bridges, and frozen input data. |
| [tests/](tests/) | Selector, numerical, and lifecycle checks. |
| [tuning/](tuning/) | Candidates compiled only with `CK_INT8_CACHED_TUNING=ON`. |

The example owns its headers. The original provider manifest and the
16-bit-input grouped GEMM providers remain in `19_grouped_gemm`.

## Build and reproduce

Set `E` to the absolute path of this example in the `ck_yyc` container.

```bash
B="$E/build/cached"
python3 "$E/build_and_manifest.py" --build "$B" --jobs 2

export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$B"
export CK_INT8_TEST_VERSION=v158
export CK_INT8_CACHED_BENCH_LIB="$B/libck_tile_int8_nt_cached_benchmark.so"
export LD_LIBRARY_PATH="$B:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:${LD_LIBRARY_PATH:-}"
GATES="$E/results/validation"
mkdir -p "$GATES"
export CK_INT8_K_LIFETIME_OUTPUT="$GATES/k_parallel_lifetime.json"
ulimit -c 0
python3 "$E/tests/test_acceptance_host.py"
python3 "$E/tests/test_scaled_api.py" --output "$GATES/scaled_validation.json"
python3 "$E/tests/test_scaled_epilogue.py" --paired --output "$GATES/epilogue_validation.json"
python3 "$E/tests/test_cached_api.py"
python3 "$E/tests/test_lifetime.py"
python3 "$E/tests/test_k_parallel_lifetime.py"
python3 "$E/benchmarks/run_scaled.py" --output "$E/results/scaled_run"
```

The scaled runner times one formal round, with 10 warmups and 50 repetitions per
sample. The reported time is the median of three samples. Weight preparation is
recorded separately. Throughput counts two operations per INT8 multiply-accumulate.
Use a new output directory for each run.

The [raw INT32 case list](benchmarks/data/cases_int8_nt_1178.csv) can also be
used with the [CSV driver from example 19](../19_grouped_gemm/run_grouped_gemm_device_args_cases.py).

Historical reports and probes are available in the
[v158 source snapshot](https://github.com/tracybj/composable_kernel_das/tree/aed28615896065b68b602a1bb9ccbfcce03f362c/example_hcu/ck_tile/24_grouped_gemm_int8_nt)
and the [experiment branch](https://github.com/tracybj/composable_kernel_das/tree/exp/int8_nt/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
