# Iteration49: large-M screening validation memory fix

The original344-shape v48-library screen stopped after132 complete shapes /792 valid records. Active shape464 has M8192,N7168,K2048,G96 on GPU7 (63.98GiB). The exception occurred in Python `(c != reference).sum()`: HIP requested an additional42GiB while only13.28GiB was free, with50.09GiB allocated. This was a validation reduction temporary, not evidence of kernel address failure or changed mathematical support.

Preserve the complete792-record prefix and original manifest. Resume with screen_large_chunked.py, requiring six valid records per completed shape and an exact original-case prefix. Compare every output element in per-group chunks of256 rows, bounding temporary memory. All original M/N/K/G, candidates, library SHA, output/reference types, HIP-event warmup/repeat/sample rules are unchanged. Do not count the interrupted shape as validated. The resume manifest records this harness change and revision.
