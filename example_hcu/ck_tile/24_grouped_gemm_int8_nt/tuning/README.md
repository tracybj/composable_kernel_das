# Offline tuning for scaled INT8 NT grouped GEMM

The runner enumerates compiled instances for each uniform, nonempty shape and
each BF16/FP16 output. The dispatcher and enumerator share the two instance
registration files under `instances/`. Eligibility checks native tile geometry,
alignment, scale/writeback requirements and prepared-weight availability. An
explicit candidate is checked again at plan creation, before binding weight
format or uploading GPU arguments.

The tuning entry points are enabled by `CK_INT8_CACHED_TUNING`. Build and run
from this example's root, with the DTK environment initialized and one device
visible. For example, on physical card 7 of the benchmark host:

```sh
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$PWD/build/offline_tuning"
export CK_INT8_CACHED_BENCH_LIB="$CK_INT8_CACHED_BUILD/libck_tile_int8_nt_cached_benchmark.so"
export LD_LIBRARY_PATH="$CK_INT8_CACHED_BUILD:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:${LD_LIBRARY_PATH:-}"
python3 build_and_manifest.py --tuning --build "$CK_INT8_CACHED_BUILD" --jobs 4
python3 tuning/run_tuning.py --pci 0000:f5:00.0 --output results/tuning_1191
```

Use `--ids` for a subset, or `--shapes` for a CSV with ID, phase, M, N, K and G
columns. Use `--resume` with the same arguments to reuse completed cases. The
manifest checks the build, runner, input list, device and timing settings.

Each case shares input buffers, output storage and prepared weights across
candidates. The default public plan is a control; if it selects a candidate
already in the catalog, the runner measures it once. The first pass measures
every legal native candidate twice, with forward and reverse ordering. The
first sample uses one timed invocation. The second uses extra repeats for the
default and candidates within 15% of the fastest first sample, then every
candidate is checked against the entire reference output and guards. The second pass
measures the top three correct candidates, those within 1.5% of the fastest
screen result, and the default, using three samples. The winner is the lowest
median among these finalists. The reference uses bounded INT8 inputs, an exact
FP32 integer dot product and the v2 scale/rounding order.

`cases/<ID>.json` records every measured instance, the raw samples, correctness
and finalist status. `results.csv` records the default, winner, runner-up,
latency, throughput and saved microseconds for each output. Static preparation
time and packed bytes are separate from the steady-state kernel timings. This
runner uses the original dimensions and therefore creates no activation padding.

Treat the result as an offline ranking for the recorded build, device and
input contract. Use a separate run to verify close winners and derive selector
rules from the measured shape families. The normal build and plan API keep the
existing selector.
