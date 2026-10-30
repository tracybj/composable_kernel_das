# M128 WASP experiment

Keep native INT8 MMAC,128-wide K stages,all-C one-barrier publication,and the approved packed-weight format. Split A rows over two producer waves,not four. Each consumer owns128x32 rather than256x32:64accumulators/lane instead of128. Two16KiB A LDS slots total32KiB instead of64KiB. Compare N64/128/256 with2/4/8consumers and M-group1/4 traversal. The five M256 controls retain the prior validated geometry/address policies in the same library.

Hypothesis:smaller LDS and accumulator state can permit more independent CTA/wave work;smaller N stripes can fill the grid for low-G,M128. Cost:more CTAs and repeated operand requests,especially repeated B for splitting M,and repeated A for splitting N. Lower resources alone do not prove a gain. Physical weight preparation remains one-time and separate. No production selector/API/default or FP16/BF16 source is changed.

CPU model verifies every A-stage byte mapping in both LDS slots. GPU full-range/FP64 guards and same-build timed comparisons are required before promotion. Native shape domain is M divisible128,N divisiblethe selected N tile,K divisible128 with supportedstrides. A complete scalar fallback exists only for diagnostic tail checks;it is not proposed as a production tail optimization. Packed-input diagnostics additionally require N%16==0,K%64==0.
