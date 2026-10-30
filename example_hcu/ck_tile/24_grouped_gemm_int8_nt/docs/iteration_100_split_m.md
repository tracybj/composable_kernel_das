# Iteration100: M128 WASP with smaller LDS and narrower N stripes

Independent CK-Tile Problem/Pipeline/Kernel/Epilogue. Two producers replace four;each consumer owns128x32 instead of256x32. Two16KiB A slots use32KiB LDS instead of64KiB. CompareN64/N128/N256 with2/4/8consumers,plus A batches2/4/8. Keep the validated all-C single-publication protocol,native INT8 MMAC and approved packed-weight format. Original FP16/BF16 and current isolated production API/defaults unchanged.

CPU A-byte address audit passes for both slots.212 full-range INT8/FP64 GPU checks PASS:all16 variants,tails,ragged/empty groups,padded/wide strides,odd/even K-stage counts,packed bytes through independent Torch layout construction,unchanged inputs and guarded outputs. All194 representative performance outputs also PASS. GPU7,14 original representative gap shapes,same library,current-policy controls,warmup2/repeat5/3samples median,rotating order per case/sample. This is development screening,not final1178 acceptance.

Actual metadata:all new variants32KiB LDS,118--135VGPR,zero private scratch/spills. Same-build original controls64KiB LDS,214VGPR raw and198VGPR packed. Resource reduction is verified;resident-CTA count is not measured and no occupancy limit is claimed.

| M | N | K | G | currentTOPS | best instance | bestTOPS | R |
|---:|---:|---:|---:|---:|---:|---:|---:|
|16384|2048|512|8|182.164|36202|212.675|1.0580|
|16384|2048|768|8|242.202|36402|256.671|0.9942|
|16384|4096|1536|4|383.001|32001|383.001|0.9716|
|128|6144|2048|8|126.680|36402|225.892|1.6557|
|128|7168|2048|8|132.160|36402|230.955|1.6130|
|4096|1024|2048|64|368.370|36202|417.367|1.1158|
|128|1536|2048|4|61.947|36402|123.970|1.8193|
|16384|1536|2048|4|429.068|36202|434.274|0.9843|
|128|3072|4096|4|124.892|36402|212.819|1.4847|
|16384|3072|4096|48|569.003|32001|569.003|0.9896|
|16384|4096|6144|8|628.292|32001|628.292|0.9870|
|128|4096|7168|12|163.985|36402|282.535|1.6130|
|16384|4096|7168|8|647.651|32001|647.651|0.9588|
|8192|6144|7168|12|644.307|32001|644.307|0.9838|

Packed M128/N128 variant36402 improves the five selected M128 shapes by roughly1.70--2.00x and exceeds their targets. Packed M128/N256 batch2 variant36202 closes the representative K512 and K2048/N1024 gaps. Seven of14 representative gaps close in this screen. Large long-K shapes still prefer the current M256 packed path;smaller resources do not provide a universal improvement. ShortK768 remains slightly under target.

Next screen all78 M128 and344 M>=256 original requirements against current selection and the promising packed variants,plus dimension-eligible raw controls. Promote only explainable M/N/K/G rules after broad numerical/performance checks and then fresh public-API acceptance. Until then formal production remains iteration99:1093/1178 target attainment,1178 numerical PASS,85 unmet. One-time preparation/storage are separate;experimental Torch packing interval is not the production create cost.
