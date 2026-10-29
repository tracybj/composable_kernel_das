# Iteration98: full1178 acceptance runner for the isolated INT8 API

## Scope

The objective remains all1178 original INT8 NT shapes meeting the frozen BLAS target. This iteration prepares acceptance of the new cached-weight API;it does not claim new throughput or replace formalv80. Physical7 HIP initialization remains unavailable in a fresh native check(error100). No new GPU job is launched and no driver/global environment changes are made.

## Runner

`cached_weights/tests/validate_all_1178.py` checks the complete frozen target table(306prefill+872decode),INT8 input/INT32 no-scale/layout contract and original M/N/K/G. It requires passing full-range boundary and reuse/refresh tests for the exact new library SHA before GPU acceptance. It also validates the actual runtime PCI identity against physical7 renderD135(currently0000:f5:00.0,confirmed by rocm-smi).

Each original shape uses10warmups/50repeats/3samples,median HIP-event GEMM time through the new public interface. Weights are prepared once per shape and reused for timing. Submit-host time,host time until readiness,the event interval around preparation,and extra packed bytes are separate columns. The preparation event interval includes allocation/metadata/host submission gaps and must not be described as pure packing-kernel time. It is a single preparation observation,not a repeated-cost median.

All output elements are compared with an exact integer FP32 reference for input[-8,8],TF32disabled,in128-row per-group chunks. Only one group's FP32 weights and bounded reference/reduction temporaries are live;the original group count is never reduced. Output guards and sampled input immutability are checked. Separate full-range INT8/FP64 boundary tests remain mandatory.

JSONL records are flushed/fsynced one complete shape at a time. CSV always retains1178 rows;unmeasured rows remainPENDING with empty performance cells. Explicit resume checks the frozen libraries,target CSV,compiled context,test source hashes,original ordered dimensions,phase,dtype/layout,target values,timing parameters,arithmetic and numerical flags. It records a hash of the pre-existing prefix. Errors are preserved separately;there is no automatic restart or configuration change.

## CPU evidence

`cached_weights/tests/test_acceptance_host.py` passes with the actual1178-shape table and frozen artifacts. Plan-only imports neither Torch nor bindings/HIP. Fifteen deliberate corruptions of resume records are rejected. CSV checks preserve all original dimensions/phases,do not mark unmeasured rowsPASS,and only fill the synthetic validated row in a temporary directory. These are CPU harness tests;they are not kernel numerical or performance results.

The old formalv80 library and all nine frozen public-context source hashes remain unchanged. No FP16/BF16 interface,layout,default route or provider is edited.

## Run after GPU recovery

```sh
python3 cached_weights/tests/test_acceptance_host.py
# Use the existing physical7 process-local HIP/CUDA/LD_LIBRARY_PATH launcher.
python3 cached_weights/tests/test_cached_api.py
python3 cached_weights/tests/test_lifetime.py
python3 cached_weights/tests/validate_all_1178.py --output results/cached_api_v97_all_1178
# Only after inspecting a terminal interrupted run:
python3 cached_weights/tests/validate_all_1178.py --output results/cached_api_v97_all_1178 --resume
```

The current development draft still misses60 large-M targets and prior formalv80 has14 small-M misses. Development screening and formal acceptance have different timing conditions;do not combine their pass counts into a delivered1178 result. Once GPU validation is available,the new frozen-interface run determines the actual remaining cases before further pipeline promotion.
