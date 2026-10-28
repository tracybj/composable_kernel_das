# Iteration 20: CTA-local K partition

2026-10-28; physical GPU7, gfx936; INT8 NT -> INT32 without scale.

Method: 16x16 output CTA, 2/4/8 waves partition K32 chunks, register lookahead and LDS partial-output integer reduction. One kernel, no global scratch or atomics. Development-only instances; production selector unchanged.

Compile issue corrected: CK constant indices require decltype(i)::value for ext_vector indexing.

| M | N | K | G | best v19 ms | best local-K ms | target ms |
|---:|---:|---:|---:|---:|---:|---:|
|4|4096|6144|8|0.240672|0.317184|0.187976|
|8|4096|6144|8|0.242304|0.357824|0.189496|
|8|4096|7168|8|0.264832|0.415264|0.216848|
|8|6144|7168|12|0.593824|0.917632|0.471056|
|16|7168|3072|12|0.318656|0.581600|0.265272|

All 25 outputs and guards pass. All local-K variants lose to v19 on the five tested shapes. Do not promote. This only rejects the direct-load 16x16 CTA arrangement; it does not reject K partitioning paired with coalesced LDS staging.

Timing: warmup2/repeat5, median of 3 HIP-event samples, input [-8,8]. Screening only.

Exact BLAS symbol ISA is saved in build/v20/blas_exact_kernel.isa; static instruction summary in docs/blas_small_m_isa_summary.json. Raw vendor disassembly is not published to GitHub.

Raw screening records:
```json
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.2544960081577301, "samples_ms": [0.2552315890789032, 0.25436800718307495, 0.2544960081577301], "tops": 6.3286365379915255, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.24067199230194092, "samples_ms": [0.23948760330677032, 0.24128000438213348, 0.24067199230194092], "tops": 6.6921485985763, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6202, "ms": 0.782943844795227, "samples_ms": [0.782943844795227, 0.7852479815483093, 0.7795199751853943], "tops": 2.057124207191696, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6204, "ms": 0.39932799339294434, "samples_ms": [0.40089577436447144, 0.39932799339294434, 0.39868777990341187], "tops": 4.033307863831961, "bad_elements": 0, "guard": true}
{"id": 1019, "M": 4, "N": 4096, "K": 6144, "E": 8, "candidate": 6208, "ms": 0.3171840012073517, "samples_ms": [0.31996801495552063, 0.3171840012073517, 0.31670379638671875], "tops": 5.077849859605936, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6101, "ms": 0.25779157876968384, "samples_ms": [0.25779157876968384, 0.25811201333999634, 0.2570880055427551], "tops": 12.495464310251606, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6102, "ms": 0.2423039972782135, "samples_ms": [0.2423039972782135, 0.24294400215148926, 0.2420479953289032], "tops": 13.294149119221455, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6202, "ms": 1.0084158182144165, "samples_ms": [1.004639983177185, 1.0084158182144165, 1.0179197788238525], "tops": 3.1943424664874507, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6204, "ms": 0.4232639670372009, "samples_ms": [0.4232639670372009, 0.4221437871456146, 0.42524799704551697], "tops": 7.610441055373099, "bad_elements": 0, "guard": true}
{"id": 1020, "M": 8, "N": 4096, "K": 6144, "E": 8, "candidate": 6208, "ms": 0.3578239977359772, "samples_ms": [0.35830381512641907, 0.35756799578666687, 0.3578239977359772], "tops": 9.002262264077668, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6101, "ms": 0.28230398893356323, "samples_ms": [0.28300759196281433, 0.28230398893356323, 0.2817597985267639], "tops": 13.312232668750642, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6102, "ms": 0.2648319900035858, "samples_ms": [0.2648319900035858, 0.26633578538894653, 0.26476800441741943], "tops": 14.190492560770757, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6202, "ms": 1.2103996276855469, "samples_ms": [1.2118395566940308, 1.2103996276855469, 1.2100478410720825], "tops": 3.104839342346796, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6204, "ms": 0.4826880097389221, "samples_ms": [0.4826880097389221, 0.48521581292152405, 0.4802877902984619], "tops": 7.785767013422793, "bad_elements": 0, "guard": true}
{"id": 1090, "M": 8, "N": 4096, "K": 7168, "E": 8, "candidate": 6208, "ms": 0.41526398062705994, "samples_ms": [0.415039598941803, 0.4154878258705139, 0.41526398062705994], "tops": 9.049897316702431, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6101, "ms": 0.631775975227356, "samples_ms": [0.6316795945167542, 0.631775975227356, 0.634911835193634], "tops": 13.38404307152873, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6102, "ms": 0.5938237905502319, "samples_ms": [0.5940159559249878, 0.5933119654655457, 0.5938237905502319], "tops": 14.239437689360688, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6202, "ms": 2.8535356521606445, "samples_ms": [2.8535356521606445, 2.851743459701538, 2.8549115657806396], "tops": 2.963242059932732, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6204, "ms": 0.9784000515937805, "samples_ms": [0.9815998077392578, 0.9784000515937805, 0.9724160432815552], "tops": 8.642392087189615, "bad_elements": 0, "guard": true}
{"id": 1153, "M": 8, "N": 6144, "K": 7168, "E": 12, "candidate": 6208, "ms": 0.9176318049430847, "samples_ms": [0.9175357818603516, 0.918175995349884, 0.9176318049430847], "tops": 9.214716423788797, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6101, "ms": 0.31865599751472473, "samples_ms": [0.31750360131263733, 0.3189759850502014, 0.31865599751472473], "tops": 26.535564778156328, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6102, "ms": 0.3444480001926422, "samples_ms": [0.3444480001926422, 0.344512015581131, 0.34118399024009705], "tops": 24.54860199295947, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6202, "ms": 0.9481280446052551, "samples_ms": [0.9534079432487488, 0.9463998079299927, 0.9481280446052551], "tops": 8.91832797491025, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6204, "ms": 0.58160001039505, "samples_ms": [0.58160001039505, 0.5819839835166931, 0.5814080238342285], "tops": 14.538715118413563, "bad_elements": 0, "guard": true}
{"id": 579, "M": 16, "N": 7168, "K": 3072, "E": 12, "candidate": 6208, "ms": 0.588096022605896, "samples_ms": [0.588096022605896, 0.5816320180892944, 0.5882560014724731], "tops": 14.378122855740644, "bad_elements": 0, "guard": true}
```
