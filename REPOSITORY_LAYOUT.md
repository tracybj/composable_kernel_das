# Repository layout and provenance

- `main`: complete source snapshot of the user-maintained CK tree, with the
  reorganized INT8 NT example at `example_hcu/ck_tile/24_grouped_gemm_int8_nt`.
- `exp/int8_nt`: original optimization-only delivery history and experiments,
  preserved before replacing the root of `main` with the full source tree.
- Source snapshot: remote `/root/workspace/yyc/GroupGemm/composable_kernel`,
  source revision `81559d413ed2ef5698ae6ca0cd64f018b5941396`, followed by the
  two-core refactor. No upstream Git history or tracking remote was imported.
- Original CK/Hygon copyright, licenses and upstream provenance remain intact.
- Build outputs, core dumps, vendor disassembly, credentials and large runtime
  artifacts are excluded. Experimental source and development probes from this
  example are archived on `exp/int8_nt`; production code, tests, contracts and
  reports remain on `main`.
- The original raw INT8 API and the additive cached-weight INT8 API are distinct.
  FP16/BF16 ABI and dispatch remain unchanged. Latest INT8 performance uses
  immutable prepared weights; packing time/storage are reported separately.

The v120 stopping point kept the accepted v112 production policy. Experimental
M192/stagger candidates are historical investigations, not production defaults.
The new verification report is recorded separately in the example's `docs/`.
