# Iteration106:integrate M64 packed routes into the isolated INT8 API

Add routes8(M64xN256,A-batch2)and9(M64xN64,A-batch2)under `cached_weights/m64/`. UniformM64 native plans become eligible. Route8 requiresN>=4096,K>=4096,G<=12;all other native M64 usesroute9. The selection uses only dimensions and availability. Ragged/empty and unsupported N/K preserve the existing provider. Both routes explicitly bind owned packedB using existing format1;no new weight layout or activation packing is introduced.

Build in `build/cached_v106/` passes. Exactly8 new C API symbols are exported,zero old grouped-GEMM ABI owners are added. C structs/tags/offsets and all9 old public-context source hashes are unchanged;original INT8 host45 tests pass. Same-process loading preserves all4 old C entrypoint addresses. This is ABI/loader evidence,not new FP16/BF16 numerical or performance evidence.

55 full-rangeINT8/FP64 boundary checks PASS with route coverage0/1/3/4/5/6/7/8/9,including M64 native packed routes,wide padding,odd stage count,ragged/empty fallback,input immutability,packed-byte checks,guarded output and nondefault preparation/compute streams.20 shared-weight tests PASS across M64/128/256/512/2048,including explicit source-address refresh,two generations,two streams,and queued-work destruction. CPU input/CSV/resume corruption checks for1178 shapes PASS.

The dimension policy was calibrated on iteration105's92-case screen and is not independent acceptance. A fresh full1178 run is required. Existing formal v103 record remains1178 numericPASS,1117 reach BLAS,61 remain until the new public-API run finishes and is audited. Pack preparation and persistent bytes are reported separately under the approved reusable static-weight contract. Full target completion remains unproven.
