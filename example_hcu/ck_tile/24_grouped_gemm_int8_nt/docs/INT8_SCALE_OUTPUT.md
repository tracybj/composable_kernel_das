# INT8 NT fused scale and output conversion

The integer GEMM pipelines remain Cooperative and WASP. The scaled kernel
wrappers reuse their INT32 accumulators and fuse scaling/conversion into the
epilogue. No global INT32 intermediate or separate conversion kernel is used.
The existing FP16/BF16-input providers are unchanged.

## Contract

For group `g`, output row `m`, and output channel `n`:

```
acc[g,m,n] = sum_k int32(A[g,m,k]) * int32(B[g,n,k])
combined_scale = FP32(scaleA[g,m] * scaleB[g,n])
D[g,m,n] = RNE_output(FP32(acc[g,m,n]) * combined_scale)
```

Scale modes independently select a device FP32 scalar or an outer vector.
The activation vector has M entries. The weight vector has N entries
(per-channel). A NULL pointer has value 1. Scale mode is not a K-axis scale:
per-K/block quantization is not supported. There are no zero points, bias,
nonzero beta, activation functions or output requantization in this interface.

| Setting | Output | Entry |
|---|---|---|
| Original no-scale interface | INT32 | Original `plan_create_v1` |
| v2, no scale, output_type=0 | INT32 | `plan_create_v2` |
| v2, output_type=1 | BF16 | FP32 scale in GEMM epilogue |
| v2, output_type=2 | FP16 | FP32 scale in GEMM epilogue |

All groups in one plan must use the same output type. INT32 rejects non-NULL
scales and non-scalar scale modes. Explicit BF16/FP16 identity conversion is
allowed with NULL scales. FP16 overflow produces infinity;it is not clamped.
Conversion uses round-to-nearest, ties-to-even. BF16 NaNs are quieted and retain
a NaN classification;payload equality is not required by numerical gates.

`ck_tile_int8_nt_cached_group_desc_v2` has `abi_version=2`, the exact struct
size, and zero reserved fields. Strides count elements of the corresponding
tensor. Existing alignment/stride domains apply (16-byte A/D base alignment,
A row stride a multiple of 16,output row stride a multiple of 4).

Weights retain the existing create/destroy and INT8-only packing contract.
Plans retain the existing run/query/destroy entry points. Scale tensors are
borrowed and are neither packed nor cached by value. They must remain valid until all runs
complete. Update their contents only with correct compute-stream ordering.
Preparation owns its packed-weight buffer and records a readiness event.
Plan creation owns and uploads its descriptor and scale-pointer arrays once.
Caller-owned A, B, output and scale storage stays live until work completes,
and the prepared weights handle outlives its plans. Reuse a plan for the same
addresses and metadata. No additional caller workspace is required. Run uses
the supplied stream and performs no device allocation, GPU descriptor readback
or synchronization. The first submission on a stream waits on the preparation
event. Preparation and plan destruction have their own lifetime rules.

The [cached extension header](../../../../include/ck_tile/ops/gemm/grouped_gemm_int8_nt_cached.h)
is separate from the device-args v1 ABI. The latter still accepts caller-owned
`GemmTransKernelArg` arrays and the expected instance ID. Scale pointers and
the prepared-weight handle are expressed in the cached extension. Use
`weights_create_v1 → plan_create_v2 → plan_query_v1 → run_v1` for scaled INT8.

Example initialization:

```cpp
ck_tile_int8_nt_cached_group_desc_v2 d{};
d.struct_size=sizeof(d);d.abi_version=2;
d.a=a_int8;d.c=d_bf16;d.m=M;
d.a_row_stride=K;d.c_row_stride=N;
d.output_type=CK_TILE_INT8_OUTPUT_BF16_V2;
d.scale_a=activation_scalar;d.scale_b=weight_channel_scales;
d.scale_a_mode=CK_TILE_INT8_SCALE_SCALAR_V2;
d.scale_b_mode=CK_TILE_INT8_SCALE_OUTER_VECTOR_V2;
// Build G descriptors,then call plan_create_v2(weights,groups,G,&plan).
// Use the existing cached_run_v1(plan,stream) and plan_destroy_v1(plan).
```

## Reference calculation and BLASLt behavior

The archived 1178-case INT32 acceptance uses inputs [-8,8], TF32 off, and a full
PyTorch FP32 matrix product cast to INT32. That input domain makes the integer
result exactly representable in FP32. The old boundary gates separately use
full-range INT8 and FP64 references. Neither test validated scaling or 16-bit
output conversion.

The new gate uses full-range INT8 [-128,127] and computes the integer dot in
FP64, casts to INT32, then follows the FP32 operation order above and casts to the
selected output type. It compares all output elements.

On the installed `/workspace/GroupGemm/baseline/hipblaslt-install` library on
gfx936, native GroupedGemm scalar-scale probes execute INT8 to BF16 and FP16
with `HIPBLAS_COMPUTE_32F`, C/D both the selected 16-bit type, and non-NULL scalar
scaleA/scaleB. The underlying product is an integer MMAC result. Nontrivial
random-input probes identify `acc * FP32(scaleA*scaleB)` as the matching
operation order. Multiplying `acc*scaleA` then `*scaleB` can differ after
conversion. The test-only BLAS reference also checks the actual loaded library
path and the device argument scale pointers.

DTK can fold the last FP32 multiply and FP16 conversion into
`v_fma_mixlo_f16`. The resulting single-rounding operation differs at some
halfway cases from BLASLt's FP32 multiply followed by FP16 conversion. A
register constraint preserves that FP32 rounding boundary without adding a
separate kernel. A minimum code-generation experiment confirms the explicit
`v_mul_f32` followed by `v_cvt_f16_f32` sequence after this constraint.

The BLAS reference transposes the row-major problem without copies:
`D^T = B * A^T`. Consequently,BLAS scaleA is the CK weight scale and BLAS
scaleB is the CK activation scale. These mappings must not be confused.

The inspected native GroupedGemm outer-vector and scaleAlphaVec probes did not
find a supported algorithm for the tested INT8-to16-bit configurations.
Ordinary Gemm probes also did not find a supported scaled algorithm at the
tested shape. This is a measured limitation of these configurations,not a
claim that every library version or shape lacks the feature. Per-channel CK
tests therefore use the independent exact-dot reference;they are not labeled
as a successful native per-channel BLAS comparison. Scalar native comparisons
require bitwise identical finite outputs,without relaxing a tolerance.

The negative grouped probes also supplied scaleAlphaVec together with both
scalar scales,and both outer-vector mode flags;neither combination found an
algorithm. Passing a vector pointer in scalar mode executed but used only its
first element. That behavior must not be labeled per-channel support.

## Validated result on physical GPU7

The final scaled gate passes36 configurations and328 complete-output checks,
plus7 invalid-descriptor checks. It covers20 production route/legacy-instance
families,the13 added low-N shapes,and the special boundary/rounding cases.
All6 native BLASLt comparisons have zero mismatched16-bit output patterns.
The same final library passes65 old INT32 boundary checks and32 lifetime checks.
The initial v122 validation is archived in [validation identity and coverage](https://github.com/tracybj/composable_kernel_das/blob/aed28615896065b68b602a1bb9ccbfcce03f362c/example_hcu/ck_tile/24_grouped_gemm_int8_nt/docs/scale_v122_validation.json).
The optimized v123 scalar-A epilogue, current validation, and performance
evidence are described in [scale epilogue optimization](https://github.com/tracybj/composable_kernel_das/blob/aed28615896065b68b602a1bb9ccbfcce03f362c/example_hcu/ck_tile/24_grouped_gemm_int8_nt/docs/SCALE_EPILOGUE_V123.md).

## Build and gates

Use the existing DTK toolchain and physical GPU7:

Follow the [build and validation steps](../README.md#build-and-reproduce).


To enable native BLAS cross-checks, compile [blas_scale_reference.cpp](../tests/blas_scale_reference.cpp)
as a shared test bridge against the explicit installed BLASLt library, then
pass `--blas-reference /path/to/blas_scale_reference.so`. Set
`CK_INT8_BLAS_LIBRARY` to the expected library path to enforce its identity.
The bridge is a test oracle only and never participates in CK dispatch.

The boundary gate covers the historical route and legacy-instance fixtures,
the 13 added low-N shapes, and padded, ragged, and empty groups. It also checks
NaN, infinity, rounding ties, and the original INT32 output. The
[release report](CURRENT_INT8_NT_STATUS_AND_FLOW.md) covers the full run.
