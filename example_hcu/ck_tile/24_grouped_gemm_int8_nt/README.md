# INT8 NT grouped GEMM v157

This example provides cached signed INT8 grouped GEMM in NT layout, with INT32
accumulation and alpha=1, beta=0. The v1 API writes INT32. The v2 plan also
supports fused FP32 scales with BF16 or FP16 output. Expert weights can be
prepared once and reused across calls.

The card 7 full run covered all 1191 shapes. Both outputs passed all numerical
checks. Against the frozen workbook target, BF16 reached 1162/1191 and FP16
reached 1190/1191. See the [current implementation](docs/CURRENT_INT8_NT_STATUS_AND_FLOW.md),
[scale contract](docs/INT8_SCALE_OUTPUT.md), and [audited CSV](docs/max_all_v157_1191.csv).

## Build and reproduce

Set `E` to the absolute path of this example in the `ck_yyc` container.
Build the production cached API and benchmark bridge with DTK.

```bash
python3 "$E/cached_weights/build_and_manifest.py" \
  --build "$E/build/v157_release" --compiler /opt/dtk/bin/aicc --arch gfx936 --jobs 2
```

Use physical card 7. Run the numerical, boundary and lifetime gates before the
full run. Use a new output directory for each benchmark.

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$E/build/v157_release"
export CK_INT8_CACHED_BENCH_LIB="$CK_INT8_CACHED_BUILD/libck_tile_int8_nt_cached_benchmark.so"
export CK_INT8_TEST_VERSION=v157
export LD_LIBRARY_PATH="$CK_INT8_CACHED_BUILD:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:${LD_LIBRARY_PATH:-}"
GATES="$E/results/max_all_v157_gates"
mkdir -p "$GATES"
export CK_INT8_K_LIFETIME_OUTPUT="$GATES/k_parallel_lifetime.json"
ulimit -c 0
python3 "$E/cached_weights/tests/test_scaled_api.py" --output "$GATES/scaled_validation.json"
python3 "$E/cached_weights/tests/test_scaled_epilogue.py" --paired --output "$GATES/epilogue_validation.json"
python3 "$E/cached_weights/tests/test_cached_api.py"
python3 "$E/cached_weights/tests/test_lifetime.py"
python3 "$E/cached_weights/tests/test_k_parallel_lifetime.py"
python3 "$E/experiments/max_all_v136/run_all_scaled.py" --output "$E/results/v157_release_run"
```

The runner measures one round with 10 warmups, 50 repetitions per sample,
and three samples summarized by the median. Weight preparation is recorded
separately. Throughput uses two operations per INT8 multiply-accumulate.
The reference target comes from [the frozen workbook extract](docs/targets_max_all_1191.csv).

## Source and validation history

`main` contains the complete CK source, the maintained INT8 NT example, and the
v157 acceptance artifacts. The files under `experiments/` in this branch support
the optional tuning build and the maintained acceptance runner. Iteration
scripts, intermediate results, and the full optimization log are on
[`exp/int8_nt`](https://github.com/tracybj/composable_kernel_das/tree/exp/int8_nt/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
Original licenses and source provenance are retained.
