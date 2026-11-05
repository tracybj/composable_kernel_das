# Conditional follow-ups, not implemented or GPU-tested

Evaluate only after the pending representative timing/profile results. Keep
factors separate and freeze any new scope before running it.

* K16 redundant A requests: `V7SmallMGeometry` has blockM64, four waves, physical
  A slab4096B (128×16 BF16). A row is blockStart+wid×32+(lane>>1); every A load
  in waves2/3 gets invalid INDEX due to the blockM64 bound. Those writes fill an
  unused upper half of the physical A slab. Existing wave-group split is
  exactly wid<2. If dynamic VMEM/control remains relevant, a new K16 candidate
  could omit these known-invalid A requests in the already separate groupB
  body, without changing geometry, LDS reservation, LGKM0 or block barriers.
  Must rederive per-wave VMEM wait counts (loader waves2requests/stage,
  other waves1) including all prologue/drain paths. Request reduction is not
  HBM reduction: invalid INDEX already avoids actual global-memory reads.
* Small common G lookup: new steady providers currently preserve the old
  scalar descriptor scan for G<=16. If profiles show this fixed cost matters,
  the internal host tile-per-group argument can be positive for all common,
  nonragged/nonempty groups as it already is for the tail providers. Test
  direct lookup versus scan as one factor on76/633/681 plus dense controls;
  variable division may offset a shorter scan. No public ABI/caller field is
  needed. CAPI-only changes avoid recompiling shared selector headers.
* Tail M64: if128-square/wide tails are still limited on M<=64, the existing
 64×128 core with the already independent vector-masked tail epilogue is an
  additional task1 candidate. It requires B-load tail masking too, a new ID,
  full precision, and a complete attribute domain; not an ID winner list.
* Buffer-depth comparison is conditional on new profile evidence. Do not
  assume two-stage LDS or two waves is faster or copy Native partial-wait
  constants. The current four-stage/three-bank lifetime proof must remain
  valid unless replaced with a new explicit proof and stress validation.

Rejected CPU hypotheses: benchmark num_cu defaults0, so current new V7 grids
are already exact (slots1), not rounded to80. Device properties are cached,
not queried from HIP on every timed public-API call. Source-level API CPU work
still exists, but no cost/bottleneck claim has been measured.
