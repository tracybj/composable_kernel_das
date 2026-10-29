# Iteration97: additive INT8 cached-weight API;16bit interface isolation

## Contract and implementation

User authorizes one-time INT8 expert-weight preparation/reuse and explicitly requires that FP16/BF16 interfaces remain unaffected. New code lives only in `cached_weights/`,built independently. Existing parent CMake,public C header/structs,selector and FP16/BF16 providers are unchanged. The INT8-only opaque handles define eight new symbols;the existing INT8 provider is linked under a private hidden name. This module defines none of the existing grouped-GEMM C entry names.

Prepare copies signed INT8 NT weights to `[G,K/64,N/16,4,16,16]` for eligible N/K. Source weights remain immutable and valid until handle destruction for raw fallback. No global pointer-keyed cache;updates require a fresh handle. Plans bind A/C and M/strides,can share weights,and track a prepare event for first use on each stream. No device allocation or host-to-device metadata copy occurs in Run. Destroy waits used streams;live plans prevent weight destruction. Activations are never prepacked.

Dimension dispatch follows the iteration96 M/N/K/G draft: raw linear transposed paths for short-K/M256 and selected K2048/N<=1024 families,raw linear original for K2048/1024<N<2048/M>=4096,otherwise packed original weights in the native domain. Ragged/empty/non-native cases use the private unchanged raw provider. The draft is calibrated on development data;its scores are not independent acceptance results.

## Verified compatibility and build

- Shared library build passes;SHA256 `e39601a17a85e8392ae83197c71bdd1af86c2c2510fc71e35ac9231d5a159ac3`.
- New API exports eight new C names and zero old entry names. Both new and old headers compile together as C11;old FP16/BF16 tags,struct sizes and offsets are unchanged.
- Source SHA checks pass for all nine frozen existing public-context files.
- Same-process old/new loading preserves all four existing public symbol addresses,both on the old library handle and in global lookup.
- Existing general selector: `SELECTOR_TABLE_PASS cases=80 registry=49`;existing quant selector:zero failures. Original INT8 selector:45 checks pass.
- New boundary and explicit reuse/refresh tests are saved and parse successfully. These are planned GPU checks,not reported as numerical passes.

## Current GPU environment failure

The first new-API boundary run stops at the first Torch GPU allocation,before any new kernel. Separate controls with Torch alone,the known v90 library,and the new library all fail HIP initialization. Native `hipGetDeviceCount` returns100 (`no ROCm-capable device is detected`);`rocminfo` returns HSA_STATUS_ERROR_OUT_OF_RESOURCES. Opening `/dev/kfd` and card7 renderD135 succeeds. Container inherits DTK_VISIBLE_DEVICES=2,3,4;per-process tests selecting physical7 do not resolve the failure. Its causal role is therefore unproven.

Host read-only inspection found repeated `failed to get hyuds net namespace` warnings and no matching hyuds process;this is evidence for administrator investigation,not a proved root cause. No driver,service,container-global environment or other GPU is changed. Detailed controls and code hashes are in `iteration_97_int8_cached_api.json`;initial log is preserved under build/cached_v97.

**GPU numerical/stream/lifetime validation is pending. No FP16/BF16 GPU regression pass or new1178 performance acceptance is claimed. Original v80 remains the formal delivered baseline.**

## Resume validation

After physical7 HIP availability is restored,use the existing process-local launcher environment and run `cached_weights/tests/test_cached_api.py` followed by `cached_weights/tests/test_lifetime.py`. The first covers full-range INT8/FP64 output,guards,padded/ragged/empty cases and routes0/1/3/4/5. The second checks immutable weights shared across M256/M512 plans,multiple streams,queued-work destruction and explicit refresh at unchanged source addresses. Then run independent1178-shape timing/numerical acceptance with the frozen new library,reporting preparation cost and extra bytes separately. Keep old FP16/BF16 entry points and defaults unchanged throughout.
