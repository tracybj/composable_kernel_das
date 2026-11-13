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

### Public integration

The existing [device-args v1 header](../../../include/ck_tile/ops/gemm/grouped_gemm_device_args.h)
retains its select, query and run contract. Its INT8 path writes INT32 with
no scales. Use the [versioned cached INT8 extension](../../../include/ck_tile/ops/gemm/grouped_gemm_int8_nt_cached.h)
for fused FP32 scales and BF16 or FP16 output. The example-local header
forwards to the public extension header for source compatibility.

| Consumer | CMake target and contract |
| --- | --- |
| Existing host descriptors or device args | `ck_tile_hcu_grouped_gemm` and its existing provider archive. Continue using `GemmTransKernelArg`, `contract_flags` and the selected instance ID. |
| Fused INT8 NT scales on gfx936 | `ck_tile_int8_nt_cached`. Call weights create, plan create v2, plan query and run on the caller's stream. Build with `CK_ENABLE_INT8` and `CK_EXPERIMENTAL_BIT_INT_EXTENSION`; the target sets both. |
| Offline tuning | Configure the cached example with `CK_INT8_CACHED_TUNING=ON`. The default build keeps this option off and uses the built-in selector. |

The existing device-args provider manifest still includes `instances.cpp`.
The cached extension target also builds `grouped_int8_nt_cached_api.cpp`,
`instances/grouped_int8_nt_cached_dispatch.cpp` and
`instances/grouped_int8_nt_scaled_dispatch.cpp`. Those files use the existing
route IDs and a shared scaled-route list for dispatch and offline enumeration.
The selector uses dimensions, device CU count and descriptor properties.
The scale contract and ownership rules are described in
[INT8_SCALE_OUTPUT.md](docs/INT8_SCALE_OUTPUT.md).
To build the optional raw INT8 CK versus BLASLt benchmark, set
`CK_INT8_BENCHMARK_HIPBLASLT_ROOT` to the installation used by that benchmark.
The scaled CK benchmark uses the public cached extension directly and has no
BLASLt link dependency.

## Build and reproduce

Set `E` to the absolute path of this example in the `ck_yyc` container.

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

The [raw INT32 case list](benchmarks/data/cases_int8_nt_1178.csv) can also be
used with the [CSV driver from example 19](../19_grouped_gemm/run_grouped_gemm_device_args_cases.py).

For scaled INT8, the same driver accepts `int8_bf16` and `int8_fp16` as input
dtype names, `layout=NT` and `backend=ck`. These modes use scalar activation
scale and per-channel weight scales, and compare all output elements and
guards against an exact reference. With the full CK build, use
`tile_example_grouped_gemm_device_args_benchmark`. With the standalone example:

```bash
cmake --build "$B" --target tile_example_grouped_gemm_device_args_benchmark_int8
"$B/tile_example_grouped_gemm_device_args_benchmark_int8" \
    --dtype=int8_bf16 --layout=NT --backend=ck --include-args-build=0 \
    --m=16384 --n=1536 --k=2048 --groups=4 --warmup=10 --samples=3 --gemm-repeat=50
```

The full 1191-shape Python run additionally uses randomized operands and the
frozen BLASLt target for each shape.

Historical reports and probes are available in the
[v158 source snapshot](https://github.com/tracybj/composable_kernel_das/tree/aed28615896065b68b602a1bb9ccbfcce03f362c/example_hcu/ck_tile/24_grouped_gemm_int8_nt)
and the [experiment branch](https://github.com/tracybj/composable_kernel_das/tree/exp/int8_nt/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
