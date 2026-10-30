# INT8 NT grouped GEMM — two-core organization

This example implements signed INT8 × INT8 → INT32, NT layout, alpha=1,
beta=0, no scales. Static expert weights may be packed once and reused;
dynamic activations are never packed. FP16/BF16 interfaces are unchanged.

## Production layout

| Layer | Files | Responsibility |
| --- | --- | --- |
| Cooperative pipeline | `grouped_int8_nt_pipeline.hpp` (`pipeline/cooperative.hpp`) | Compute waves move both A and B through LDS, prefetch the next K tile in registers, and execute MMAC. |
| WASP pipeline | `pipeline/wasp.hpp` | Dedicated producer waves stage A into two LDS slots; consumers load B directly into registers and execute MMAC. One shared main loop for BM=32/64/128/256. |
| Problem and instances | `cached_weights/core_problem.hpp`, `cached_weights/instances.hpp` | Compile-time geometry and policy: BM/BN/BK, producer/consumer waves, A-read batch, weight address format, record stride, transpose and output mapping. |
| Selection | `cached_weights/selector.hpp`, `include/ck_tile/ops/gemm/grouped_gemm_selector.hpp` | Select an instance from M/N/K/G, layout, strides, alignment and device CU count. Never uses workload case IDs, phase labels or performance targets. |
| Kernel / epilogue | `cached_weights/core_kernel.hpp`, `cached_weights/core_epilogue.hpp` | One grouped launch wrapper and output mapping for the WASP instances. |
| API / packing | `cached_weights/cached_api.cpp`, `pack_kernel.hpp` | Weight ownership, one-time INT8-only packing, plan/stream lifetime and launch. Packing is not a third GEMM pipeline. |
| Compatibility | Previous m32/m64/m128/wide_n headers | Thin aliases only. Old public INT8 ABI remains available; the unused legacy direct kernel remains for ABI compatibility. |

The v120 stopping point retained the v112 production selector and accepted
implementation; v113–v120 experiments did not improve the remaining eight
cases and are not enabled. The refactor preserves this production policy,
rather than promoting the v120 experimental M192 candidate.

The reorganized version has completed one new full round on physical GPU7:
1178/1178 numerical PASS, 1170/1178 at or above the frozen BLAS target;
prefill 298/306, decode 872/872. Minimum ratio is 0.981198. Median speedup
versus v112 is 1.000034x, so performance is effectively preserved.
No OOM skips or missing metrics. The same eight prefill cases remain below
the target; this refactor does not claim to close them.

See [report](docs/REFACTOR_TWO_CORE_v121.md) and
[1178-case CSV](docs/current_refactor_v121_1178.csv).
BLAS numbers are the frozen tuned targets, not a new BLAS measurement.

## Build

```bash
R=/workspace/GroupGemm/composable_kernel
E=$R/example_hcu/ck_tile/24_grouped_gemm_int8_nt
python3 "$E/cached_weights/build_and_manifest.py" \
  --build "$E/build/refactor_v121" --compiler /opt/dtk/bin/aicc \
  --arch gfx936 --jobs 2
```

The cached target is also available from the example/root CMake build.
Link the old raw API owner only once; the cached-weight API has eight separate
INT8-only C symbols and does not define an old FP16/BF16 owner.

## Reproduce validation and performance

Use physical GPU 7 in `ck_yyc`. Run the boundary and lifetime gates first,
then the full suite, using a fresh output directory (one formal round):

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$E/build/refactor_v121"
export CK_INT8_CACHED_BENCH_LIB="$CK_INT8_CACHED_BUILD/libck_tile_int8_nt_cached_benchmark.so"
export CK_INT8_TEST_VERSION=v121
export LD_LIBRARY_PATH="$CK_INT8_CACHED_BUILD:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:$LD_LIBRARY_PATH"
ulimit -c 0
python3 "$E/cached_weights/tests/test_cached_api.py"
python3 "$E/cached_weights/tests/test_lifetime.py"
python3 "$E/cached_weights/tests/validate_all_1178.py" \
  --output "$E/results/refactor_v121_new_run"
```

The acceptance runner freezes source/library/target hashes, validates all
outputs and guards, and uses 10 warmups, 50 repetitions, three samples and
the median. Static-weight preparation time and storage are reported separately.
BLAS columns are the frozen tuned Excel targets, not a new BLAS measurement.
No result is substituted with a faster sample from another run.

Full output equality uses inputs [-8,8] with an exact integer FP32 reference
and TF32 disabled. Boundary tests use full-range INT8 and FP64 reference,
nonuniform/empty groups, positive padded strides, guards and full input
immutability; lifetime tests cover cached-weight reuse and stream ordering.
These checks do not prove correctness for every possible tensor value.

## Repository branches and provenance

`main` contains a complete CK source snapshot plus the maintained example.
`exp/int8_nt` retains the optimization experiments and their history.
This repository does not track an upstream remote or import upstream Git
history. Original license/copyright and source provenance are retained.
Historical experimental recipes should be run from their recorded revision
on `exp/int8_nt`, because production compatibility headers no longer contain
separate experimental main loops.
