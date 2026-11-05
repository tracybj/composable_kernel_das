# P6f independent tail and steady pipelines — 实验用，不合入

Implemented default-off instances5013/5016 (independent128×128×16 and
128×256×16 N tails),5014/5015 (64×128×32 /64×128×16 independent steady/no-global
tail), and5017 (limited matched small-M NoSLC cache control). CMake/provider,
query masks/registry, public candidate IDs and default selector entry agree.
Public v1 structs, functions and GemmTransKernelArg semantics are unchanged.
No implicit hot-path GPU allocation/readback/synchronization was added.

Tail contracts: common positive M<=128, N>=128,N%8=0,N%128!=0,K64..4096%16,
G<=1024,nonragged/nonempty,gfx936 BF16NN and byte/tile limits. B loads and C
stores both bound complete eight-BF16 vectors. The original accepted epilogue
is untouched. Tail256 uses8B A loads in all lanes; tail128 uses16B A loads;
both NoSLC. Per-wave store slabs skip complete invalid16-row M fragments after
retaining the original global/LDS/block drain. Tail contracts remain separate
from existing no-column-predicate paths.

Steady contracts: common M<=64,N%128=0,G<=1024,nonragged/nonempty,gfx936 BF16NN,
K32 requires128<=K<=4096%32; K16 requires64<=K<=4096%16. Both retain4-stage LDS,
3operand banks,12-phase and all necessary LGKM0/block barriers. A full-cycle
steady body has no per-stage bounds branches, followed by one phase switch and
at most5 no-global-load tail steps. Prologue4 is optional only when S>4, so
source-level stage loads are exactly S rather than S+5. Request count is not
HBM bytes or a performance result. The limited5017 control is only four shapes,
620/621/76/358; its shader differs from5003 only in SLC machine bits.

Formal four-target CMake build passed. CPU-only old13 GPU shader bodies are
byte-identical to this run's frozen cache binary; all5 new gfx936 shader bodies
have zero scratch/spill. Static compiler reports in both configured targets
also show zero scratch/spill for these five bodies. New resources: tail128
VGPR123/SGPR48/LDS32KiB; tail256124/48/48KiB; K32steady116/70/48KiB;
K16steady80/60/32KiB; smallNoSLC80/47/32KiB. Actual dispatch checks remain pending.

246 finite accuracy jobs passed874 full-output CK-versus-per-group hipBLASLt
comparisons,6,806,867,656 elements, max_abs=0, guards passed. This includes all16
N1232 shapes under both widths, M/N/K/G/ragged/empty/disable boundaries; K32
beforeK16 across short stages4..28 and48/64/128; repeated20-run stress at stages
4/5/16/17; specified representatives; all15new shapes with affected/fallback
checks; fresh current-best route baseline precision; and the four cache controls.
Official selector/query/smoke passed, including the actual7 C-API byte-range
cases. CPU selector checks77288+6720, cache2880 and old14316 byte-identical
selections passed. These are not1193-shape GPU performance results.

One K16S64 precision command returnedrc0 and3 zero-error full comparisons but
was excluded on a post-job hymgr unregister event. Fourth bounded recovery
passed known original5000 probes and entered epoch4; the failed physical record
was retained and replaced with a new key. Other accepted accuracy records are
from healthy epoch3/4. Cumulative recovery1056.69s includes waits/diagnosis.
No device reset/service restart/card switching was used. No formal new-instance
performance result is claimed in this checkpoint; finite timing/profile plans
are running after accuracy. Eight additional max-K/full-M-min-stage corners
are predeclared for final acceptance and remain pending here.

Binary SHA3411731442660e51e3681e5341bbf852409b47b8e80fcffe15a92e52e45dbaac.
Exact commands, environments, per-job SHA/epoch and all logs are in the compressed
journals. Source/build manifests and CPU lifetime/address/ABI audits are included.
The user-owned readfirstlane patch remains unchanged and unstaged; it is an
explicit existing workspace prerequisite, not copied into this commit. Both
1193-row shape CSV hashes remain fixed. No default global enable, push or full
1193/1178 GPU sweep. Performance/adoption and the long-term BLAS target are pending.
