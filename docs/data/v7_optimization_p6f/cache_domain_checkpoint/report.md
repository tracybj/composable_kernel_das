# P6f cache-domain diagnosis checkpoint — 实验用，不合入

Completed the original12-case, predeclared attribute domain. Eight N>4096 cases
improve12.2%–38.6%; all four N4096 NoSLC/M01=1 controls are within0.4% and have no
confirmed benefit. No regression exceeds2%. Domain equal-call paired total is
−28.33%, median per-shape change−14.27%; these are diagnostic mixed-shape totals,
not workload-frequency weights. All arms/rounds for one comparison use one GPU,
one binary and one environment epoch. ID781 was fully repeated in epoch2:
84.482246→52.782368ms, paired−36.21%, stable direction.

The proposed policy is refined to common/nonragged/nonempty BF16 NN gfx936,
1024<M<=2048,N>4096,N%256=0,3072<=K<=4096,K%16=0,G<=1024 and original byte/tile
bounds. This selects NoSLC/M01=5; N4096 keeps the original fallback. The refined
8-case domain and all four boundary controls were frozen for a fresh official
executable verification before adoption. That verification is still pending;
this checkpoint does not claim it has run.

Seven clean profiler captures are retained. In epoch2, original/M01=5 SLC/
M01=5 NoSLC HBM reads are81.29/60.57/31.18GB; original→combined L2 hit55.83→79.09%,
TCP–TCC latency746.1→344.9cycles. All use43008 CTAs,512threads,64KiB dynamic LDS,
zero scratch and VGPR212/SGPR48. VMEM474instructions/wave is unchanged; MMOP
matches the padded geometry. The cache behavior changes, while request count
and compute geometry stay fixed. Profiler model times are not formal timing.

Two profiler captures returned rc0 but overlapped hymgr transitions and are
excluded; both raw records/logs are retained. The initial regression pre-gate
also saw a service event before any HIP work. Three bounded recoveries passed
known original5000 full-output probes, reaching epoch3. Cumulative recovery
812.81s includes observation/wait/diagnosis time. The first recovery ran an
extra immediate pre-probe gate after its two60-second-separated observations;
the runner was corrected to reuse the final healthy observation as the probe
gate. Later recoveries used at most three observations separated by60seconds.
No GPU reset/service restart/card switching to evade recovery was used.

The optional M01=8/Native profiler completion was stopped after the second
profiler-window event. Their missing profiles are not inferred. Original and
candidate clean profiles satisfy the required minimum; frozen Native same-card
three-round performance diagnostics for781–784 are complete in the earlier
checkpoint. All174 formal CK jobs ended outside the later event windows.

Live audit:250 attempted/248 accepted jobs,174 formal timing jobs/29 complete
contrasts,52 full-output jobs/76 comparisons/39,096,735,744 elements, max_abs=0.
Two raw profile records are invalid. No1193-shape GPU suite was run. Tail and
decode source is integrated with default-off options and zero-spill static
compilation, but GPU validation is pending. The full P6f task is not complete.
