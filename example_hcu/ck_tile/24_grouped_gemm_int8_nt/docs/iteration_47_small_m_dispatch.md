# Iteration47: reusable no-LDS epilogue and long-K small-M selection

Add a typed optional epilogue parameter to the existing CK-Tile kernel and launch factory. The default parameter preserves all existing paths. The new epilogue transposes four MMAC accumulator components across the four lane groups using ds_bpermute, then writes a contiguous16-byte INT32 vector without an LDS output exchange or CTA barrier. Misaligned C base/row stride uses scalar stores. The common layout guard requires N divisible by4; masked M/N tails and padded strides remain supported.

Register instance8734: BM16,BN32,BK512, two compute waves, masked packed input, no-LDS output. Select it for uniform nonempty M<=16 and K>=4096, at every G. Selection depends on dimensions and request properties, never case ID/phase. Lower-K small-M routes and all M>16 routes retain their previous rules. This brings the registry to24 INT8 plus20 floating-point entries.

Host selector tests: original80 and INT8 33 PASS. All1178 CPU replay/relabel/reverse cases PASS. Public boundary tests:28 full-range INT8 / independent FP64 exact-reference profiles PASS, covering K16/N4 tails, ragged/empty fallback, padded strides, nondefault stream, full input clones, output guards and error contracts. These checks do not establish full1178 throughput.

Next measure the default public API on all560 M<=16 targets, original dimensions and fixed workbook references, GPU7, warmup10/repeat50, three HIP-event samples/median. No claim of full1178 target completion. Contract: gfx936,INT8 NT to INT32,no scale.
