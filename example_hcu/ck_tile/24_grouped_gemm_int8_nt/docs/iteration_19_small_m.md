# Iteration 19: small-M packed LDS

2026-10-28; gfx936 physical GPU7; INT8 NT -> INT32, no scale.

Experimental candidates: 6101 = 16x128x64 / 1x2 waves; 6102 = 16x256x64 / 1x4 waves. Both masked, peeled K64, single LDS, vectorized epilogue. Production selector unchanged.

Three HIP-event samples, warmup2/repeat5, [-8,8] inputs; full output and guards verified. These are screening results, not formal acceptance.

| M | N | K | G | direct ms | best packed ms | speedup | target ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
|4|4096|6144|8|0.457728|0.240832|1.901|0.187976|
|8|4096|6144|8|0.509056|0.241984|2.104|0.189496|
|8|4096|7168|8|0.573952|0.265632|2.161|0.216848|
|8|6144|7168|12|1.171136|0.593216|1.974|0.471056|
|16|7168|3072|12|0.619936|0.317696|1.951|0.265272|

Result: useful improvement but all five still below the spreadsheet target. Keep experimental; next compare K partitioning/reduction and MMAC/load scheduling. No case-ID production dispatch.

BLAS selected kernel for M8,N4096,K6144,G8: Custom_Cijk_Alik_Bljk_I8II_BH_Bias_AI_MT96x16x256_SE_K1_LSU4_PGR2_padding_GEMVMT256_WGM1_ccyBlockSize256_gfx936

Names suggest LSU4/PGR2/GEMV structure; not yet verified from ISA. The first function-filtered dump produced only headers. Symbol data saved under build/v19; no claim of completed disassembly.

Raw probe records follow:
```json
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6004, "ms": 0.45772799849510193, "samples_ms": [0.4615035951137543, 0.4570237994194031, 0.45772799849510193], "tops": 3.5187114209646384, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.2545599937438965, "samples_ms": [0.2543675899505615, 0.2545599937438965, 0.2552638053894043], "tops": 6.327045787172586, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.24083200097084045, "samples_ms": [0.24118399620056152, 0.24063999950885773, 0.24083200097084045], "tops": 6.687702338174778, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6004, "ms": 0.5090556144714355, "samples_ms": [0.5090556144714355, 0.5090240240097046, 0.5113919973373413], "tops": 6.327845878577873, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.2560960054397583, "samples_ms": [0.2560960054397583, 0.25705599784851074, 0.25603199005126953], "tops": 12.57819490963412, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.24198400974273682, "samples_ms": [0.24128000438213348, 0.24233600497245789, 0.24198400974273682], "tops": 13.311728636221119, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6004, "ms": 0.5739516019821167, "samples_ms": [0.5739516019821167, 0.5760639905929565, 0.5666880011558533], "tops": 6.547758331924816, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6101, "ms": 0.28227201104164124, "samples_ms": [0.28227201104164124, 0.28303998708724976, 0.2820480167865753], "tops": 13.313740778378483, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6102, "ms": 0.2656320035457611, "samples_ms": [0.26524800062179565, 0.26633602380752563, 0.2656320035457611], "tops": 14.147754539496153, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6004, "ms": 1.1711360216140747, "samples_ms": [1.1619516611099243, 1.1732478141784668, 1.1711360216140747], "tops": 7.220098014188157, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6101, "ms": 0.6318399906158447, "samples_ms": [0.6318399906158447, 0.6347519755363464, 0.6297599673271179], "tops": 13.382687056193362, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6102, "ms": 0.5932160019874573, "samples_ms": [0.593504011631012, 0.5927680134773254, 0.5932160019874573], "tops": 14.254026923870446, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6004, "ms": 0.6199358105659485, "samples_ms": [0.6208955645561218, 0.6190080046653748, 0.6199358105659485], "tops": 13.639665139332172, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6101, "ms": 0.3176957964897156, "samples_ms": [0.3176957964897156, 0.3184959888458252, 0.31747201085090637], "tops": 26.615765639422705, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6102, "ms": 0.34275200963020325, "samples_ms": [0.3458560109138489, 0.34275200963020325, 0.3423359990119934], "tops": 24.670072315908264, "bad_elements": 0, "guard": true}
```
