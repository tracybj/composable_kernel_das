# Iteration75: matched long-M BLAS/CK comparison and counters

M16384,N4096,K6144,G8,physical GPU7,INT8 NT to INT32,no scale. Replay frozen matmul algorithm2979 and public CK instance19003 on identical A/B/C allocations. Three unprofiled samples,5 warmups/20 repetitions,reverse backend order in the middle sample. All six full-output/guard/full-input-clone checks PASS. No retuning or target change.

| Backend | Median ms | TOPS | R to frozen target |
|---|---:|---:|---:|
|BLAS|10.377902|635.684|0.998590|
|CK|13.202150|499.697|0.784968|

The gap remains in fresh same-input execution. BLAS matmul backend launches8 independent expert GEMMs per complete call; CK launches one grouped kernel. Both include all8 independent input/weight/output matrices and the same2*G*M*N*K work. The 637TOPS workbook target is retained.

Capture DTK hipprof PMC type3 separately,2 warmups/1 measured complete call. Both profiled outputs/input clones/guards also PASS. Exact kernel-name matching finds24 BLAS launches and3 CK launches; aggregate last8 BLAS rows and last1 CK row to represent the measured complete G8 call. Other3157-row capture entries are reference/initialization work and are excluded. Profiler overhead is much larger for the8-launch BLAS path; do not rank performance using profiled event times.

| Counter per complete G8 call | BLAS | CK |
|---|---:|---:|
|SQ_INSTS_VALU|430276608|464486400|
|SQ_INSTS_LDS|100663296|109051904|
|SQ_INSTS_VMEM_RD|25427968|25165824|
|SQ_INSTS_VMEM_WR|8388608|2097152|
|SQ_LDS_BANK_CONFLICT|0|0|
|SQ_WAIT_INST_LDS|114369265|22087338|
|TCC_HIT|269055799|251023919|
|TCC_MISS|66630428|84530577|

Reported workgroup size768,LDS65536B,scratch0 for both; arch_vgpr reports256 for BLAS and216 for CK. These are per-kernel resource fields,not additive counters. CK's VALU/LDS instruction counts are higher,VMEM-read count is similar,and its accumulated LDS wait count is lower. CK has more TCC misses in this capture. This supports testing CTA access order/cache reuse next; it does not prove a sole bottleneck. TCC requests are not assumed to equal HBM bytes,wait counts are not percentages of runtime,and this is one controlled profile capture.

No default promotion or current-public CSV replacement. Next compare M-group CTA orders1/2/4/5/8/16 with the same typed WASP pipeline and unchanged math; screen both internal orientations and preserve original group counts. All1178 fixed-target work remains active;263 public73 gaps are recorded separately.
