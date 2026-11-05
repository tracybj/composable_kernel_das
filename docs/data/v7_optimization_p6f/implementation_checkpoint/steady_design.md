# Independent steady/tail design (CPU reasoning; GPU validation pending)

K32 uses64×128×32, four12KiB LDS stages, three operand banks; K16 uses
64×128×16, four8KiB stages and the same bank/phase schedule. The raw pipeline
uses `G::kStageK` to advance K; inherited framework shape encoding is not the
source for the actual K stage. `RunGemmV7Uniform` passes the complete raw K.

Let S=K/stageK. Prologue loads0..3 and materializes operands0/1. It waits until
stage2 is globally complete, drains both LDS operand reads and the block, then
loads stage4 only when S>4. A steady step t computes operand bank t%3, reads
stage t+2 into bank(t+2)%3, and loads t+5 into LDS slot(t+5)%4. All steady
reads/loads are in range by the enclosing loop condition, with no per-stage
bounds branch. Twelve steps return both rings to their original phase.

After the last steady step, at most five compute steps remain. A single phase
switch enters a bounded fallthrough sequence, including the11→0 wrap. Its
global-load body is absent at compile time. The remaining operand reads still
check t+2<S. A tail step waits for at most one remaining stage of global loads
when t+4<S, otherwise zero. It retains the original complete LGKM wait and the
block barrier. Every LDS slot is read into registers before another wave can
overwrite it; a bank is not overwritten before its MMAC consumer. Epilogue
reuse retains the final global/LDS/block drain.

At S=4 the optional stage4 load is absent. The first tail step waits VMEM0,
making stage3 ready before the following look-ahead read. At S=5 the first tail
step leaves only stage4 outstanding, then the second waits VMEM0. At S=17 a
complete12-step steady cycle issues5..16, followed by five tail computes.
The CPU models cover S4..1024,524794 stages, including every phase residue.
These models establish indexing/lifetime intent; they do not prove GPU memory
ordering or numerical correctness.

Source-level global stage requests become exactly S, compared with S+5 in the
original body. This is a request count, not an HBM-byte or performance claim.
The oldest expanded tail prototype spilled; it was rejected before GPU work.
The compact loop prototype increased K32 VGPR216/K16 VGPR144. The current
fallthrough version compiles at K32 VGPR116/SGPR70 and K16 VGPR80/SGPR60,
zero scratch/spill. Actual official executable resources and dispatches must
still be checked before adoption.
