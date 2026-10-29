# Iteration56: fresh frozen-BLAS/CK comparison on identical inputs

M2048,N4096,K8192,G128,physical GPU7,INT8 NT to INT32,no scale. Use the original frozen algorithm6950 and bridge binary; do not retune or change spreadsheet targets. Same physical A/B/C allocations for every path, three samples with reversed order in the middle round, warmup5/repeat20. Every one of12 outputs passes full reference comparison, output guards and full A/B clones.

| Backend | Candidate | Median ms | TOPS | R to frozen target |
|---|---:|---:|---:|---:|
|BLAS|6950|29.237331|601.703|1.000437|
|CK|9003|36.565388|481.116|0.799940|
|CK|9505|36.187698|486.137|0.808289|
|CK|9507|36.510521|481.839|0.801142|

BLAS replays at601.7 TOPS, essentially its frozen601.4 TOPS target, while the tested CK paths deliver481-486 TOPS. Thus this representative gap remains under fresh same-input measurement; there is no evidence here for lowering the fixed target or attributing the whole gap to an environment change. No public default promotion. Next evaluate the structured B/CACHE_SWIZZLE and invariant-address mechanism already present in the BF16 WASP source.

CK library SHA256: 222b911c51f3f1f98f0389a34bbb229739a46f1c7939388b0a49f346e110186b
