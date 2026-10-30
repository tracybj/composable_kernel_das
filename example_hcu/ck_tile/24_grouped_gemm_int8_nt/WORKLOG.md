
## Iteration97: isolated INT8 cached-weight API

Additive new module builds;old public context hashes/ABI/4symbol addresses unchanged;80-case general selector and quant selector pass. New GPU boundary attempt fails before any kernel;Torch-only and known-library controls fail identically,nativeHIP error100 and HSA initialization resources error. Numerical/lifetime1178 acceptance pending environment recovery. See docs/iteration_97_int8_cached_api.md and JSON. Existing FP16/BF16 code/defaults and formalv80 baseline unchanged.


## v121 — two production cores and full-source main

- Source base: v120 stopping point 81559d4; keep v112 accepted dispatch and geometry.
- Unify five WASP implementations into pipeline/wasp.hpp; cooperative core retained.
  Shared Problem/Kernel/Epilogue, named instances, old public ABI intact.
- ABI and 45 selector checks PASS; 65 full-range INT8 boundary + 32 lifetime checks PASS.
- GPU7 formal full round: 1178/1178 numeric PASS, 1170 >= BLAS target; same eight prefill
  gaps; no skips/missing metrics. Median speedup vs v112 1.000034x.
- ID62 ABBA diagnosis did not reproduce a stable 3% regression; formal data unchanged.
- Restore M32 alias default batch=2; both rebuilt .so files byte-identical to the fully
  tested artifacts. Preserve original run-source manifest separately.
- Preserve experiments on exp/int8_nt; main contains complete CK snapshot without
  upstream Git history and a production example without the archived experiments.
- Report: docs/REFACTOR_TWO_CORE_v121.md; full CSV: docs/current_refactor_v121_1178.csv.
