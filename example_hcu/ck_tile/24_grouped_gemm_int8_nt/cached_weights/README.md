# INT8 NT cached-weight API

The [C header](ck_tile_int8_nt_cached.h) defines the v1 INT32 output path and the
v2 plan descriptor for fused scale with BF16 or FP16 output. The ABI, validation
rules, and rounding behavior are documented in the [scale contract](../docs/INT8_SCALE_OUTPUT.md).

1. Create the weights handle from aligned NT INT8 descriptors on the loading stream.
2. Create a plan with per-expert activation and output pointers, row counts, and strides.
3. Run the plan on a stream. The run waits for weight preparation when required,
   reuses uploaded descriptors, and allocates no device memory.
4. Destroy the plans before destroying their weights. Plan destruction waits
   for work submitted through the plan.

Original weights must remain valid and immutable for the lifetime of the
weights handle because raw fallbacks borrow those buffers. Prepare a new
handle when changing weights. Scales remain caller-owned and must outlive
submitted work. Use the same current HIP device for all operations on a
handle, and serialize destruction with operations using that handle.

| Location | Responsibility |
| --- | --- |
| [cached_api.cpp](cached_api.cpp) | C entry points, weight and plan lifecycle, descriptor validation. |
| [instances/](instances/) | Shape-based selector, problem aliases, and INT32 or scaled launch dispatch. |
| [kernels/](kernels/) | Packing, cached kernel wrappers, K-parallel decode, and fused epilogues. |
| [tests/](tests/) | Numerical, rounding, selector, and lifecycle validation. |

The packed native weight layout is `[G,K/64,N/16,4,16,16]`. Preparation time
and extra weight storage are tracked separately from steady-state GEMM. The
[example README](../README.md) has the build and test commands.
