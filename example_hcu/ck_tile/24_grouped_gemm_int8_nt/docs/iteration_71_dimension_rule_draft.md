# Iteration71: dimension-only rule draft

The draft below uses only M/N/K/G. Zero means retain the existing dimension-based default route. Nonzero values are kernel instance handles, not test case IDs. This is development-data analysis and is not installed in the public selector. A same-build public validation is still required, including guards, padded stride/native fallback and all1178 original shapes. Any regressions and target misses remain visible.

```cpp
unsigned draft_instance(int m,int n,int k,int g) {
    if (m <= 2048) {
        return 18411;
    } else {
        if (k <= 768) {
            return 18811;
        } else {
            return 18401;
        }
    }
}
```

```json
{
  "scope": "Dimension-only draft rules trained/evaluated on344 development shapes; no held-out evidence or public acceptance",
  "method": "Depth<=3 tree, minimum12 shapes per leaf; minimize sum log latency ratio vs current default; split requires >=0.5% mean-log improvement",
  "features": [
    "M",
    "N",
    "K",
    "G"
  ],
  "tree": {
    "attribute": "M",
    "threshold": 2048,
    "count": 344,
    "left": {
      "action": 18411,
      "count": 146
    },
    "right": {
      "attribute": "K",
      "threshold": 768,
      "count": 198,
      "left": {
        "action": 18811,
        "count": 22
      },
      "right": {
        "action": 18401,
        "count": 176
      }
    }
  },
  "predicted": {
    "total": 344,
    "ge1": 99,
    "min_R": 0.7587955956334363,
    "median_R": 0.9498076525128778
  },
  "oracle": {
    "total": 344,
    "ge1": 109,
    "min_R": 0.7587955956334363,
    "median_R": 0.9542826236407698
  },
  "legacy": {
    "total": 344,
    "ge1": 11,
    "min_R": 0.45609053335500704,
    "median_R": 0.6110435596582857
  },
  "regressions_gt3pct": 1,
  "worst_vs_default": 1.030564863410013
}
```
