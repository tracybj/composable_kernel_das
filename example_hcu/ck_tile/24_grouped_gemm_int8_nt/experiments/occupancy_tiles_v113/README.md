# Untested remaining-gap candidates

M128/N192 uses2 producers and6 consumers(512threads),32KiB LDS,and balanced8waves perCTA. A4/A2 variants test shorter live A ranges. M256/N320 uses10 consumers withA2 prefetch orA4 without prefetch. M256/N384/A4 disables A prefetch to avoid the known register overflow of the prefetched A4/C12 shape. All retain format1,full barriers and guarded N-stripe epilogues.

Measured hardware:80CUs,4SIMDs/CU,196608 registers/CU,64KiB LDS/CU,2560max threads/CU. Per-SIMD register partitioning may limit a block pair even when total registers fit;actual residency and benefits remain hypotheses. No performance claim until resource,numerical and paired timing gates pass. This prototype is not uploaded or selected in production by preparation alone.
