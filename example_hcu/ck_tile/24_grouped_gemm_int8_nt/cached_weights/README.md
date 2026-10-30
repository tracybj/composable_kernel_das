> 当前维护版已归一为两套核心流水，并完成新一轮 1178 条回归。
> 构建、复测及最终 CSV 见 [example README](../README.md) 和
> [v121 整理报告](../docs/REFACTOR_TWO_CORE_v121.md)。以下保留原 API 设计记录。

# INT8 NT cached-weight extension

This is an additive,standalone signed INT8×INT8→INT32/no-scale interface.
It defines new `ck_tile_int8_nt_cached_*_v1` symbols and does not own or modify
the existing FP16/BF16 grouped-GEMM ABI,structs,layouts or default selector.
The existing INT8 provider is compiled under a hidden private symbol for raw
fallback. No old C API entry names are defined by this shared library.

1. Create weights from aligned NT `int8_t` descriptors,using the loading stream.
2. Create a plan binding the activation/output pointers and per-group M/strides.
3. Run the plan on a stream. Preparation is waited on before that stream's first
   use;later launches allocate no device memory and reuse argument descriptors.
4. Destroy plans after use,then weights. Plan destruction waits its own used
   streams;weights cannot be destroyed while plans remain live.

The original source weights remain valid and immutable for the handle lifetime,
because raw fallback borrows them. Weight changes require a new prepare. There
is no global cache keyed by an address,and dynamic activation A is not packed.
All calls use the same current HIP device as creation. Destroy must not race
with other operations on the same handle.

For the calibrated native N/K family,weights are packed to
`[G,K/64,N/16,4,16,16]` by a vector16 copy kernel. Families which never use the
packed path(K<512,non-native N/K or K>=16384)avoid allocating an unused copy.
Preparation time and extra storage are separate from steady-state GEMM timing.
The M/N/K/G rule comes from iteration96 development screening;new-API full-domain
acceptance is required before treating those screening numbers as delivered.

Build this directory independently with the existing DTK compiler:

```sh
cmake -S cached_weights -B build_cached -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc -DCMAKE_BUILD_TYPE=Release
cmake --build build_cached --target ck_tile_int8_nt_cached_benchmark -j2
```

The C header uses fixed INT8 inputs and INT32 outputs. It has no FP16/BF16 or
quantized-output parameter,and does not reinterpret or extend the old structs.

The two M128 packed routes(6/7)use the same weight format and lifetime contract.
Route6 is128x256/A-batch2;route7 is128x128/A-batch4. Selection uses M/N/K/G
from iterations101/102. Updated public-API acceptance is required;development
screen counts do not replace the prior formal1178 result.

Routes8/9 add M64 packed INT8 kernels using the same format1 and lifetime contract.
Route8 is64x256/A2 forN>=4096,K>=4096,G<=12;route9 is64x64/A2 otherwise.
Ragged,empty,or non-native N/K retain the existing provider. Integration106
requires new boundary/lifetime and full1178 acceptance;iteration105 is screening.

Routes10/11 add the M32 packed paths from iteration110. Route12 is the validated
M256/N384/A2 kernel from107/111,renamed in a private namespace;spilling107
variants and the unhelpful109 mask are not integrated. Old routes remain the
base policy. Device CU count is stored from the existing weights-create query.
For old route1/5,K>=1024,M>=2048,the integer CTA-wave model selects12 when
ratio>=1.47,or baseline>=24waves andratio>=1.44. All new routes bind owned
format1 weights. This version requires fresh boundary/lifetime/full1178 tests.
