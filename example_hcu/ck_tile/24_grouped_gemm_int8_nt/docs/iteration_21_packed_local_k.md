# Iteration 21: packed LDS plus CTA-local K partition

2026-10-28; gfx936 physical GPU7; INT8 NT -> INT32, no scale.

Each wave owns a contiguous K partition and a private LDS operand region. Per-wave global transfer uses continuous 16-byte vectors; each wave computes a 16xBN partial result. Reuse LDS for partial outputs after a block barrier, then sum partitions and vector-store INT32 in the same kernel. No global scratch or atomics.

Candidates: 6302 = BN128 / split2; 6304 = BN96 / split4; 6306 = BN64 / split4. Experimental supported domain: K divisible by split*64, positive common K/N. Do not use on unsupported K; no production selector changes.

The first compile attempt tried to modify immutable CK pointer arrays; fixed by constructing a per-partition UniversalGemmKernelArgs value.

| M | N | K | G | v19 ms | best v21 ms | relative speedup | target ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
|4|4096|6144|8|0.241184|0.237280|1.016|0.187976|
|8|4096|6144|8|0.242144|0.240608|1.006|0.189496|
|8|4096|7168|8|0.265088|0.263552|1.006|0.216848|
|8|6144|7168|12|0.593760|0.566016|1.049|0.471056|
|16|7168|3072|12|0.318464|0.276576|1.151|0.265272|

All 25 full outputs/guards pass on [-8,8] data. Three HIP-event samples, warmup2/repeat5. Screening only: further verification and coverage required before public registration. Improvement over v19 is shape-dependent and still below target on these samples. Next inspect load/MMAC schedule and LDS resource costs, then expand shape coverage.

Records:
```json
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.2549756169319153, "samples_ms": [0.2549756169319153, 0.25462400913238525, 0.25571179389953613], "tops": 6.31673238163033, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.24118399620056152, "samples_ms": [0.24131159484386444, 0.24067199230194092, 0.24118399620056152], "tops": 6.6779419918917915, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6302, "ms": 0.2417599856853485, "samples_ms": [0.2415039986371994, 0.2420479953289032, 0.2417599856853485], "tops": 6.662031896776409, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6304, "ms": 0.237280011177063, "samples_ms": [0.23734378814697266, 0.237280011177063, 0.23705598711967468], "tops": 6.78781464991642, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6306, "ms": 0.24028781056404114, "samples_ms": [0.24028781056404114, 0.23878400027751923, 0.2412480115890503], "tops": 6.702848272741417, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.2562876045703888, "samples_ms": [0.2562876045703888, 0.2571520209312439, 0.25612780451774597], "tops": 12.568791523880734, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.24214398860931396, "samples_ms": [0.2415359914302826, 0.24214398860931396, 0.24230380356311798], "tops": 13.302933888634628, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6302, "ms": 0.24284780025482178, "samples_ms": [0.2433599978685379, 0.24211199581623077, 0.24284780025482178], "tops": 13.26437986516636, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6304, "ms": 0.24060781300067902, "samples_ms": [0.2415359914302826, 0.24054400622844696, 0.24060781300067902], "tops": 13.387867300846583, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6306, "ms": 0.24185578525066376, "samples_ms": [0.24089601635932922, 0.24220800399780273, 0.24185578525066376], "tops": 13.318786104956981, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6101, "ms": 0.28422361612319946, "samples_ms": [0.28422361612319946, 0.2844160199165344, 0.2836800217628479], "tops": 13.222322744535827, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6102, "ms": 0.2650880217552185, "samples_ms": [0.2650880217552185, 0.26492801308631897, 0.2661439776420593], "tops": 14.176786861649354, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6302, "ms": 0.2635520100593567, "samples_ms": [0.2629440128803253, 0.2635520100593567, 0.2641279995441437], "tops": 14.25941082048135, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6304, "ms": 0.2804799973964691, "samples_ms": [0.2813119888305664, 0.2804799973964691, 0.28038400411605835], "tops": 13.398803547077149, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6306, "ms": 0.2672320008277893, "samples_ms": [0.2680320143699646, 0.2672320008277893, 0.266975998878479], "tops": 14.063047735146837, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6101, "ms": 0.6297599673271179, "samples_ms": [0.6349755525588989, 0.6297599673271179, 0.629472017288208], "tops": 13.426888501484923, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6102, "ms": 0.5937600135803223, "samples_ms": [0.5925759673118591, 0.5937600135803223, 0.5945919752120972], "tops": 14.24096717630537, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6302, "ms": 0.5679360032081604, "samples_ms": [0.5686718225479126, 0.5679360032081604, 0.5668478012084961], "tops": 14.888502958494081, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6304, "ms": 0.5692158341407776, "samples_ms": [0.568863570690155, 0.569919764995575, 0.5692158341407776], "tops": 14.85502749016772, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6306, "ms": 0.5660158395767212, "samples_ms": [0.5660158395767212, 0.5665917992591858, 0.5656960010528564], "tops": 14.939011018354835, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6101, "ms": 0.3184639811515808, "samples_ms": [0.31689560413360596, 0.3184639811515808, 0.31942400336265564], "tops": 26.55156427242958, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6102, "ms": 0.3424319922924042, "samples_ms": [0.3424319922924042, 0.34329578280448914, 0.339680016040802], "tops": 24.69312755327962, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6302, "ms": 0.29977601766586304, "samples_ms": [0.2986239790916443, 0.29977601766586304, 0.2998400032520294], "tops": 28.206782283114215, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6304, "ms": 0.3115200102329254, "samples_ms": [0.3115200102329254, 0.3107197880744934, 0.3115519881248474], "tops": 27.143414824869865, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6306, "ms": 0.2765759825706482, "samples_ms": [0.2765759825706482, 0.27663999795913696, 0.2764800190925598], "tops": 30.572853020019853, "bad_elements": 0, "guard": true}
```
