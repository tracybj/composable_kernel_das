# P6f initial measured conclusions — 实验用，不合入

Task1 is implemented, full-output validated and timed on all16 N1232 shapes.
Uniform tail256 is recommended within the validated attribute contract, with
experiments still default off. Three-round equal-call total drops79.655%; every
shape beats frozen Native (R1.098..1.689), no>2% regression. ID1465.268581→
0.747846ms; frozen Native.821168ms, fresh same-card single.819679ms. Tail128 is
also faster than frozen Native on all16 but tail256 is the chosen simple route;
small width differences within2% remain uncertain rather than per-ID dispatch.

ID146 actual profile has equal MMOP12042240 before/after. VMEM instructions
22067200→1646400 (13.4×fewer), non-MMAC VALU112251200→3512320 (32.0×fewer),
scratch132B/lane→0. HBM reads895.15→828.25MB fall only7.5%; L2 hit69.9→33.5%
actually declines. The large time gain is supported by eliminating spill and
instruction/control work with ordinary scheduling, not simply cache-hit ratio
or padding. Physical grid80 persistent CTAs→980 ordinary512-thread CTAs. ID147
new MMOP doubles because its M padding differs, yet time improves; do not claim
MMOP equality for all M. Actual new VGPR allocation124 can round up compiler123.

Task2 bounded cache policy is freshly verified on8active+4boundary cases. Active
8 improve11.67..39.66%, all frozen R1.066..1.200. Four N4096 controls keep5000 and
are within noise. Recommend default-off CACHE_GRID only for common/nonragged/
nonempty BF16NN gfx936,1024<M<=2048,N>4096%256,K3072..4096%16,G<=1024 plus
byte/tile bounds. This is a complete tested attribute domain, not ID dispatch.
Counter factor evidence and invalid earlier profiles are in prior checkpoints.

Task3 independent K32/K16 steady/no-global-tail implementations are numerically
correct but do not justify broad adoption.24representatives include all15new
and unsupported M128/256 fallbacks; all regressions are retained. K32 on620/621
beats the historical-current-best5003 fresh controls by10.99/13.79%, but matched
old5008→new5014 contrasts show only−1.41%/0%, within2% and observed variation.
Do not attribute those gains to the new tail refactor. Existing K32 geometry
under current conditions accounts for the main difference. LowCTA76/633/681
show no stable newK32 benefit; new15 smallK32 mostly regress. K16mostlyflat or
regresses. Fallback negatives are labelled as fallback, not actualnewkernel use.

Profiles confirm requested changes:76 K16VMEM/wave106.5→96.5, branches103.5→81,
old drain258.5; dense K16branches183.5→82. K32dense VMEM209/211→194/196 and
branches111→75, but no stable matched time gain. HBM changes are small. The old
drain adds substantial branch work, but reducing branches did not establish it
as the only/main cause of timing loss. SGPR allocation rises in the newflows.
Keep all LGKM0 and block barriers. Do not presume less loading automatically wins.
SmallNoSLC four-only controls improve6205.56%/62110.45%,76/358within noise;
HBM reductions are correspondingly modest. No broad smallM cache scan/adoption.

Frozen algorithms were replayed onGPU4 in oneepoch for seven representatives,
three rounds at single and20launch.76 Native single.023840ms versus per-launch
repeat20.017092ms (frozen.017048ms) demonstrates material measurement-scope
sensitivity. Historical R remains frozen repeat20; do not replace it with a
single-launch number. An explicit CK multiple-launch diagnostic is prepared
but has not run in this checkpoint. Formal CK remains one launch/sample.

Software issue retained: a filter with a demangle-only '<' captured no dispatch
while benchmark returnedPASS; no environment event. New raw-symbol substring
filters and new keys produced all15needed decode/isolation profiles. Runtime
requires actual .perf capture. This failed log is distinct from MKFD/service
exclusions. Native three-round resume grouping and shape-contiguous execution
were corrected without changing frozen task sets/protocols; source versions and
CPU-only controller SIGSTOP/SIGCONT are journaled; GPU children finished naturally.

All work is GPU4 serial under the global lock with fresh per-command gates.
Whole-output/guard evidence, complete three-round medians/CVs and all negatives
are included. See summary.json for exact counts; vendor perf/ISA/binaries stay
onsite. User readfirstlane diff unchanged/unstaged; both1193 shape CSV hashes
fixed. This is not a fresh1193 GPU replay. A predeclared159-case longK/moderateCTA
study of the existing5008 route is still running separately; no broad policy
claim for that region is made here. No push and no default global enable.
