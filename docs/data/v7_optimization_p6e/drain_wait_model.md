# K16 no-global-load drain wait refinement (before formal GPU measurements)

All earlier mapping probes used conservative VMEM0 at every suppressed-prefetch step. Source backup:prototypes/gemm_pipeline_v7_small_m_drain_vmem0.hpp. No performance data was collected for that prototype.

Refined wait budget: last loaded stage=min(kt+5,n-1). The next iteration reads stagekt+3, so two loads may remain for each loaded stage after it. Budget=2*max(0,last-(kt+3)): steadyVMEM4, first suppressed prefetchVMEM2, remainingtailVMEM0. Exhaustive CPU count identity for n=4..256 passed. This is a scheduling model, not hardware correctness proof.

Required LGKM0 and barrier remain unchanged. Final selector-based full-output accuracy,20-repeat stress,performance and profiler will validate this source; old probes are not evidence for the refined wait.
