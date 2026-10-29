# Iteration78: paired confirmation of record1024

M16384,N4096,K6144,G8,physical GPU7,INT8 NT to INT32,no scale. Identical physical A/B/C allocations compare frozen BLAS algorithm2979,current public v73 default19003,v77 record0 control22000,and record1024 candidate22002. Reverse backend order in the middle of three samples;5 warmups/20 repetitions per sample. Complete eight independent groups included for every backend. No profiler and no BLAS retuning.

All12 full-output checks PASS with intact output guards and full input clones. Inputs[-8,8],TF32-disabled exact FP32 reference; separate v77 full-range checks are documented in iteration77.

| Backend | Median ms | TOPS | R to fixed BLAS target |
|---|---:|---:|---:|
|BLAS|10.379311|635.598|0.998455|
|CK|13.190015|500.156|0.785691|
|RECORD0|13.205142|499.583|0.784790|
|RECORD1024|11.475702|574.873|0.903062|

Record1024 is1.1494x public CK throughput in this paired run. Record0 matches public CK,so the benefit survives same-build control and alternate timing order. The candidate reaches about90.3% of the fixed target,with remaining work required. This proves a representative gain,not all1178 completion. Next screen all344 original M>=256 shapes with both mappings and record0/1024 plus the same-library public-default instance; retain all original dimensions and group counts. Public defaults and formal CSV are unchanged.
