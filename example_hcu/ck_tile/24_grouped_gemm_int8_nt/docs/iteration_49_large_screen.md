# Iteration49: completed large-M screening

All344 original M>=256 shapes and2064 candidate output checks PASS. Original dimensions/group counts are unchanged. Compare old shape-rule control,8602,9001,9002,9003,9004. Full-output comparison uses bounded chunks after the132-shape prefix to avoid the validation-only memory spike documented in iteration_49_screening_oom.md.

The fastest candidate per shape reaches the fixed target on12/344 shapes. This is an oracle screen, not a deployable default selection or acceptance of the full1178 workload. Median best ratio is 0.8120256313653058, minimum 0.5877254055559461. Results show that the current split-role family does not close the large-M gap; continue with operand-specific data paths and compute/register scheduling.

| M | Shapes | Best reaches fixed target |
|---:|---:|---:|
|256|38|4|
|2048|108|4|
|4096|88|4|
|8192|72|0|
|16384|38|0|

Timing: physical GPU7,gfx936,INT8 NT to INT32,no scale;3 HIP-event samples,warmup2/repeat5,median. Frozen v48 library SHA256: 6f5f10d0e7557aab67a700fbfb473999d7d71a6fb1bfb8ed61cf47bade31c081
