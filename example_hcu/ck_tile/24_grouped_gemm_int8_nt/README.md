# INT8 NT grouped GEMM — two-core organization

This example implements signed INT8 × INT8 grouped GEMM in NT layout,
alpha=1,beta=0. The v1 API retains INT32 output without scales. The v2 plan
also supports FP32 scalar/outer-vector scales with fused BF16 or FP16 output.
Static expert weights may be packed once and reused;dynamic activations are
never packed. Existing FP16/BF16 input interfaces are unchanged.

See [scale semantics and validation](docs/INT8_SCALE_OUTPUT.md) for the v2
contract,the BLASLt reference and the limits of the current comparison.

## Production layout

| Layer | Files | Responsibility |
| --- | --- | --- |
| Cooperative pipeline | `grouped_int8_nt_pipeline.hpp` (`pipeline/cooperative.hpp`) | Compute waves move both A and B through LDS, prefetch the next K tile in registers, and execute MMAC. |
| WASP pipeline | `pipeline/wasp.hpp` | Dedicated producer waves stage A into two LDS slots; consumers load B directly into registers and execute MMAC. One shared main loop for BM=32/64/128/256. |
| Problem and instances | `cached_weights/core_problem.hpp`, `cached_weights/instances.hpp` | Compile-time geometry and policy: BM/BN/BK, producer/consumer waves, A-read batch, weight address format, record stride, transpose and output mapping. |
| Selection | `cached_weights/selector.hpp`, `include/ck_tile/ops/gemm/grouped_gemm_selector.hpp` | Select an instance from M/N/K/G, layout, strides, alignment and device CU count. Never uses workload case IDs, phase labels or performance targets. |
| Kernel / epilogue | `cached_weights/core_kernel.hpp`, `cached_weights/core_epilogue.hpp` | One grouped launch wrapper and output mapping for the WASP instances. |
| Scaled output | `cached_weights/scaled_kernel.hpp`, `scaled_epilogue.hpp`, `scaled_dispatch.cpp` | Reuse both integer pipelines and fuse FP32 scaling plus selected 16-bit output in the same GEMM kernel. |
| API / packing | `cached_weights/cached_api.cpp`, `pack_kernel.hpp` | Weight ownership, one-time INT8-only packing, plan/stream lifetime and launch. Packing is not a third GEMM pipeline. |
| Compatibility | Previous m32/m64/m128/wide_n headers | Thin aliases only. Old public INT8 ABI remains available; the unused legacy direct kernel remains for ABI compatibility. |

The v120 stopping point retained the v112 production selector and accepted
implementation; v113–v120 experiments did not improve the remaining eight
cases and are not enabled. The refactor preserves this production policy,
rather than promoting the v120 experimental M192 candidate.

The archived v121 INT32/no-scale build completed one full round on physical GPU7:
1178/1178 numerical PASS, 1170/1178 at or above the frozen BLAS target;
prefill 298/306, decode 872/872. Minimum ratio is 0.981198. Median speedup
versus v112 is 1.000034x, so performance is effectively preserved.
No OOM skips or missing metrics. The same eight prefill cases remain below
the target; this refactor does not claim to close them.

See [report](docs/REFACTOR_TWO_CORE_v121.md) and
[1178-case CSV](docs/current_refactor_v121_1178.csv).
BLAS numbers are the frozen tuned targets, not a new BLAS measurement.

## Maintained Python entry points

The maintained Python entry points are:

| File | Purpose |
| --- | --- |
| `cached_weights/build_and_manifest.py` | Build the cached API and benchmark bridge; freeze source/library hashes. |
| `cached_weights/tests/bindings.py` | Shared ctypes descriptors and API bindings used by the GPU tests. |
| `cached_weights/tests/test_acceptance_host.py` | CPU checks for the 1178-shape contract, CSV export and resume rejection rules. |
| `cached_weights/tests/test_cached_api.py` | Full-range INT8 numerical, stride, tail, guard and API boundary tests. |
| `cached_weights/tests/test_lifetime.py` | Prepared-weight reuse, refresh, stream ordering and destruction tests. |
| `cached_weights/tests/validate_all_1178.py` | Full-output correctness and one-round performance measurement for all shapes. |
| `cached_weights/tests/test_scaled_api.py` | Full-range INT8 gates for scale broadcasting,BF16/FP16 rounding,production route families and original INT32 APIs. |

The 17 root-level Python scripts from earlier raw-API experiments and v23–v80
validation have been removed from `main`. They are preserved at
[`exp/int8_nt` revision 40e78fd](https://github.com/tracybj/composable_kernel_das/tree/40e78fd0341731c3a8e18a19218f076f000e2e1f/example_hcu/ck_tile/24_grouped_gemm_int8_nt).
Historical documents may mention those scripts; replay them from their recorded
revision with the matching build environment. C++ kernels, pipelines, public ABI,
frozen targets and the published v121 performance results are unchanged.

## Build

```bash
R=/workspace/GroupGemm/composable_kernel
E=$R/example_hcu/ck_tile/24_grouped_gemm_int8_nt
python3 "$E/cached_weights/build_and_manifest.py" \
  --build "$E/build/scale_v122" --compiler /opt/dtk/bin/aicc \
  --arch gfx936 --jobs 2
```

The cached target is also available from the example/root CMake build.
Link the old raw API owner only once;the cached-weight API retains eight v1
C symbols and adds `ck_tile_int8_nt_cached_plan_create_v2`. It defines no old
FP16/BF16 provider.

## Reproduce validation and performance

Use physical GPU 7 in `ck_yyc`. Run the boundary and lifetime gates first,
then the full unscaled INT32 suite,using a fresh output directory (one formal
round). The following commands reproduce the archived v121 build;v122 scaled
output gates are documented in [the scale guide](docs/INT8_SCALE_OUTPUT.md).

```bash
export HIP_VISIBLE_DEVICES=7 CUDA_VISIBLE_DEVICES=7 DTK_VISIBLE_DEVICES=7
export CK_INT8_CACHED_BUILD="$E/build/refactor_v121"
export CK_INT8_CACHED_BENCH_LIB="$CK_INT8_CACHED_BUILD/libck_tile_int8_nt_cached_benchmark.so"
export CK_INT8_TEST_VERSION=v121
export LD_LIBRARY_PATH="$CK_INT8_CACHED_BUILD:/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:$LD_LIBRARY_PATH"
ulimit -c 0
python3 "$E/cached_weights/tests/test_acceptance_host.py"
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
