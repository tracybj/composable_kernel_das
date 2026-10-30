# Iteration104:wave-local packed B packets

Tested static INT8 weight format2 `[G,K/128,N/32,2,2,4,16,16]`. A wave consumes one contiguous4KiB packet, using one vector byte address and scalar K-stage offset. Same M256/N256/K128 geometry,4 producer and8 consumer waves,64KiB LDS,and the existing all-consumer publication protocol. Both formats read identical bytes and unique pages per CTA;this is not a claim of halving TLB misses. No runtime case-ID dispatch.

CPU byte-bijection/per-wave address model PASS.48 full-rangeINT8/FP64 checks PASS,including padded strides,ragged/empty groups,tails,input immutability,and output guards.16 representative original shapes,64 performance records PASS;balanced order,2 warmups,5 repeats,3-sample median. Pack preparation is recorded separately under the approved static-weight-reuse contract.

New batch4 uses197VGPR versus198 for format1,with no spills;batch2 uses181 andbatch8 uses229. All use64KiB LDS. LongK batch4 gains are mostly below0.4%;batch2 andbatch8 regress on important longK cases. M16384/N4096/K7168/G8 remains647.56TOPS versus647.66 control,below675.51 target. M16384/N4096/K6144/G8 is628.59 versus628.23 control,below636.59 target. Packet continuity alone is insufficient,so format2 is not promoted and no full344/1178 rerun is justified for this candidate.

The control is the same-build old M256 packed route,not the entire v103 selected policy. ShortK cases already served by M128 in production must not be described as gains over production. Current production remains v103:1178 numericalPASS,1117 meet frozen BLAS targets,61 remain. Next compare a distinct M64 geometry and further longK dataflow/scheduling alternatives.
