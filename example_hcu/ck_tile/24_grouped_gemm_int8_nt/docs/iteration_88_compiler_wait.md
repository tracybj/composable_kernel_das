# Iteration88: compiler-visible wait versus inline assembly

Independent 16-way factorial: asm/builtin current-B wait x early/late next-B prefetch x record0/1024 x original/transposed internal orientation. Geometry, math, MMAC memory clobbers, producer protocol, all other waits and production selector are unchanged. The builtin is __builtin_amdgcn_s_waitcnt(0x0f70); disassembly verifies vmcnt(0), rather than trusting a constant's apparent meaning.

480 full-range INT8/FP64 checks and112 representative output/guard checks PASS. All16 kernels use214VGPR,64KB LDS,zero scratch/spill. GPU7,three HIP-event samples,warmup2/repeat5,median. This is a screening experiment,not another1178-case formal run.

Generated code removes the additional compiler-inserted vmcnt(0) immediately before the first low-half MMAC: static count11 becomes10; MMAC,buffer/LDS loads and barrier counts remain unchanged. Early builtin thus no longer drains future B at that point. This verifies the code-generation explanation for part of iteration87's failure,but does not establish performance gain.

| M | N | K | G | best candidate | TOPS |
|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|30100|179.289|
|4096|1024|2048|64|30101|363.926|
|2048|1536|2048|8|30001|408.371|
|16384|3072|4096|48|30001|540.789|
|16384|4096|6144|8|30010|573.997|
|2048|6144|7168|96|30010|606.900|
|2048|4096|8192|128|30001|627.079|

early_builtin_vs_early_asm: min=1.0184,median=1.0688,max=1.0938.

early_builtin_vs_late_builtin: min=0.7599,median=0.8226,max=0.9678.

late_builtin_vs_late_asm: min=0.9915,median=1.0009,max=1.0179.

Best candidates remain late-prefetch controls. No stable material improvement justifies production promotion. Public v80 remains975/1178 fixed-target passes;all1178 exceed50%,203 full-target misses remain. Keep the original formal CSV unchanged. Next inspect the user-provided gfx938 DeepGEMM FP8 implementation for alternative task/data layouts,distinguishing packed-weight and gfx938-only assumptions from current raw-NT INT8/gfx936 requirements. Reference copies stay outside the own-code delivery repository.
