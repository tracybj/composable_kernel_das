# Iteration77: separate buffer-record stride from physical row pitch

## Evidence and independent implementation

Frozen BLAS algorithm2979 uses a power-of-two structured-buffer record stride capped at1024B. The inspected ISA prolog selects record shift8/9/10, divides physical row pitch into quotient/remainder, and forms record index=row*quotient and byte offset=row*remainder+laneK. This differs from iteration74,which merely enabled a cache bit while retaining physical row pitch as descriptor stride. Vendor ISA stays outside the own-code repository; no vendor instruction block is copied into the implementation.

Implement the address identity row*physical_pitch+laneK = (row*quotient)*record_stride + row*remainder+laneK, where quotient=physical_pitch/record_stride and remainder=physical_pitch%record_stride. B descriptor record stride is the largest power of two no larger than the cap and physical pitch. Compare caps0(original physical pitch),256,1024,4096,both internal orientations. Keep original physical allocation/strides/descriptor span,K-stage scalar offset,A producers,MMAC math,group counts,INT32 output,one-barrier protocol and packed fallback. The existing cache-swizzle guard operates on the new record stride; no unconditional cache-bit override.

## Validation and screening

176 full-range INT8/FP64 boundary checks PASS,including contiguous non-power pitches K3072/6144/7168,padded pitches,stage parity and generic tails/fallback. Guard regions and full boundary input clones are intact. All32 representative performance outputs PASS. Eight kernels use214VGPR,64KB LDS,no scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median. Boundary checks establish the tested domain,not universal safety for arbitrary buffer descriptor settings.

| M | N | K | G | best development candidate | ms | TOPS | fixed target TOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|3072|4096|48|22000|36.619255|540.459|574.971|0.9400|
|16384|4096|6144|8|22002|11.475516|574.882|636.582|0.9031|
|2048|6144|7168|96|22002|28.538940|606.796|575.891|1.0537|
|2048|4096|8192|128|22000|28.095419|626.159|601.440|1.0411|

Record1024 original orientation improves K6144 to574.9TOPS and K7168 to606.8TOPS in the representative screen. Smaller record strides regress power-of-two K4096/K8192,where original physical record stride remains faster. These results support physical-pitch-dependent selection rather than a universal replacement. Verify the gain on identical allocations with alternating BLAS/publicCK/control/record1024 timing,next screen all344 original M>=256 shapes before considering any public rule. The public all1178 CSV remains v73 and frozen targets are unchanged; no completion claim.
