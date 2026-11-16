# INT8 NT grouped GEMM

Each expert computes `C[M_e,N] = A[M_e,K] × B[N,K]ᵀ` from signed INT8 inputs.
The standalone device-args example fuses FP32 activation and weight scales
into the epilogue and writes BF16 or FP16. Expert weights can be prepared once
and reused across calls.

The v158 release on card 7 passed numerical validation and reached the frozen
BLASLt workbook target on all 1191 shapes for both scaled outputs. See the
[implementation and acceptance report](docs/CURRENT_INT8_NT_STATUS_AND_FLOW.md),
[scale contract](docs/INT8_SCALE_OUTPUT.md), and
[full benchmark CSV](docs/benchmark/v158/results.csv).

## Source layout

| Location | Responsibility |
| --- | --- |
| [instances.cpp](instances.cpp) | Raw INT8 provider entry used by `19_grouped_gemm`. |
| [instances/](instances/) | Selectors, route definitions, and kernel instance types. |
| [providers/](providers/) | HIP launchers for scaled GEMM and weight packing. |
| [kernels/](kernels/) | INT8 NT kernels, cooperative and WASP main loops, epilogues, and problem definitions. |
| [grouped_gemm_int8_nt_device_args.cpp](grouped_gemm_int8_nt_device_args.cpp) | Standalone scaled instance selection, metadata query, and launch. |
| [grouped_gemm_int8_nt_device_args_provider_sources.txt](grouped_gemm_int8_nt_device_args_provider_sources.txt) | Providers linked by the standalone device-args target. |
| [grouped_int8_nt_cached_api.cpp](grouped_int8_nt_cached_api.cpp) | Cached weights lifecycle & plan management API implementation. |
| [include/](include/) | Public & internal headers for cached weights, scaling, and K-parallel args. |
| [benchmarks/](benchmarks/) | Reproduction scripts, benchmark bridges, and frozen input data. |
| [tests/](tests/) | Selector, numerical, and lifecycle checks. |
| [tuning/](tuning/) | Candidates compiled only with `CK_INT8_CACHED_TUNING=ON`. |

## Standalone scaled device-args API

Link `ck_tile_hcu_grouped_gemm_int8_nt_device_args` and include
[grouped_gemm_int8_nt_device_args.h](include/grouped_gemm_int8_nt_device_args.h).
The API follows example 19's select, query, GPU argument construction and run
sequence. The selector uses the problem summary and selects a precompiled
instance. It reads no GPU lengths.

1. Populate the problem with the BF16 or FP16 output type and host-visible
   dimensions. For variable M, provide a total capacity and a per-group M
   upper bound. Set the ragged and empty flags as applicable.
2. Query the selected instance for the raw or packed B format. Raw B is
   row-major `[N,K]`. For packed B, allocate N*K bytes per group and call
   `ck_tile_hcu_grouped_gemm_pack_int8_nt_weights_v2` before its first use.
   Keep the caller-owned storage alive and order the consuming stream after
   the pack.
3. Build a GPU array of `GemmTransKernelArg` and a matching FP32 scale array.
   Use each group's actual M, common N/K and strides in elements, with
   `k_batch=1`. A, B and C pointers for nonempty groups must be 16-byte aligned.
   A/B strides are multiples of 16 and C strides are multiples of 4. Set the
   B stride to K for a packed view. A scale stride of 0 means a scalar and 1
   means per-row A or per-column B. A null scale pointer means unit scale.
4. Run on the caller's stream with the selected `expected_instance_id` and
   `num_cu=0`. The epilogue computes
   `float(acc) * (A_scale[row] * B_scale[column])` before the final conversion.
   The run entry point uses the GPU arrays directly and requires zero workspace.

## Build and reproduce

Set `E` to the absolute path of this example in the `ck_yyc` container.

```bash
B="$E/build/device_args"
cmake -S "$E" -B "$B" -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc \
    -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx936
cmake --build "$B" --target tile_example_grouped_gemm_int8_nt_device_args -j 4

export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
ctest --test-dir "$B" -R int8_nt_device_args --output-on-failure
"$B/tile_example_grouped_gemm_int8_nt_device_args" \
    --dtype=int8_bf16 --m=16384 --n=1536 --k=2048 --groups=4 \
    --warmup=10 --samples=3 --gemm-repeat=50
```

The standalone benchmark uses the public API and validates every output
element and the output guards. By default, its HIP event times cover the
GEMM, including the scaled epilogue. Add `--include-args-build=1` to time
the GPU argument builder as well. Weight prepacking runs before timing.

## Historical cached API and baseline

The cached-weight extension and the raw INT32 benchmark remain available for
baseline reproduction. The cached extension supports offline tuning with
`CK_INT8_CACHED_TUNING=ON`. The raw comparison benchmark accepts
`CK_INT8_BENCHMARK_HIPBLASLT_ROOT` at configuration time.

```bash
B="$E/build/cached"
python3 "$E/build_and_manifest.py" --build "$B" --jobs 2

export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$B"
export CK_INT8_TEST_VERSION=v176
export CK_INT8_CACHED_BENCH_LIB="$B/libck_tile_int8_nt_cached_benchmark.so"
export LD_LIBRARY_PATH="$B:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:${LD_LIBRARY_PATH:-}"
GATES="$E/results/max_all_${CK_INT8_TEST_VERSION}_gates"
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

The combined CSV records the planned route and pipeline separately for BF16
and FP16. `legacy_instance` identifies the selected instance when its route is
0. For other routes, `legacy_fallback_instance` reports the compatibility ID
from the plan. Pipeline fields describe the plan selection. An individual
kernel may use a generic epilogue when its vector-store conditions fail.

The [raw INT32 case list](benchmarks/data/cases_int8_nt_1178.csv) can also be
used with the [CSV driver from example 19](../19_grouped_gemm/run_grouped_gemm_device_args_cases.py).

The full 1191-shape Python run additionally uses randomized operands and the
frozen BLASLt target for each shape.

Historical reports and probes are available in the
[v158 source snapshot](https://github.com/tracybj/composable_kernel_das/tree/aed28615896065b68b602a1bb9ccbfcce03f362c/example_hcu/ck_tile/24_grouped_gemm_int8_nt)
and the [experiment branch](https://github.com/tracybj/composable_kernel_das/tree/exp/int8_nt/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
