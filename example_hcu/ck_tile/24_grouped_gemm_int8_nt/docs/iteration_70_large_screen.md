# Iteration70: completed original344 large-M screening

All344 original shapes and2408 candidate output checks PASS. Original dimensions/group counts preserved. Compare six single-barrier/A-prefetch candidates and current default instance in the same frozen library. Every output is checked with bounded row chunks.

The best per-shape choices below are an oracle screening bound, not a production selector or all1178 acceptance. Frozen workbook targets are unchanged.

| M | Shapes | Default reaches target | Best reaches target | Best median R |
|---:|---:|---:|---:|---:|
|256|38|4|26|1.0786|
|2048|108|3|40|0.9607|
|4096|88|4|34|0.9575|
|8192|72|0|9|0.9430|
|16384|38|0|0|0.9036|

Full summary:

```json
{
  "scope": "All344 original M>=256 shapes, explicit candidate screening; not default/public acceptance",
  "numeric_shapes": 344,
  "numeric_records": 2408,
  "oracle": {
    "shapes": 344,
    "ge1": 109,
    "ge05": 344,
    "min_R": 0.7587955956334363,
    "median_R": 0.9542826236407698
  },
  "same_build_default_control": {
    "shapes": 344,
    "ge1": 11,
    "ge05": 329,
    "min_R": 0.45609053335500704,
    "median_R": 0.6110435596582857
  },
  "fixed_18401": {
    "shapes": 344,
    "ge1": 64,
    "ge05": 314,
    "min_R": 0.34408714568323445,
    "median_R": 0.9347164289776312
  },
  "by_M": {
    "256": {
      "oracle": {
        "shapes": 38,
        "ge1": 26,
        "ge05": 38,
        "min_R": 0.929515725966739,
        "median_R": 1.0785676946685534
      },
      "control": {
        "shapes": 38,
        "ge1": 4,
        "ge05": 38,
        "min_R": 0.58599533437928,
        "median_R": 0.8323751486803372
      }
    },
    "2048": {
      "oracle": {
        "shapes": 108,
        "ge1": 40,
        "ge05": 108,
        "min_R": 0.8517267371649225,
        "median_R": 0.96066395199866
      },
      "control": {
        "shapes": 108,
        "ge1": 3,
        "ge05": 108,
        "min_R": 0.506817038226946,
        "median_R": 0.5991647071148849
      }
    },
    "4096": {
      "oracle": {
        "shapes": 88,
        "ge1": 34,
        "ge05": 88,
        "min_R": 0.7867383909575945,
        "median_R": 0.9574870294692647
      },
      "control": {
        "shapes": 88,
        "ge1": 4,
        "ge05": 87,
        "min_R": 0.4965612450945255,
        "median_R": 0.6078529822374603
      }
    },
    "8192": {
      "oracle": {
        "shapes": 72,
        "ge1": 9,
        "ge05": 72,
        "min_R": 0.7833574301019485,
        "median_R": 0.9429685175022069
      },
      "control": {
        "shapes": 72,
        "ge1": 0,
        "ge05": 61,
        "min_R": 0.48149253427795424,
        "median_R": 0.6105358806840684
      }
    },
    "16384": {
      "oracle": {
        "shapes": 38,
        "ge1": 0,
        "ge05": 38,
        "min_R": 0.7587955956334363,
        "median_R": 0.9036239835618378
      },
      "control": {
        "shapes": 38,
        "ge1": 0,
        "ge05": 35,
        "min_R": 0.45609053335500704,
        "median_R": 0.6102469432140465
      }
    }
  },
  "best_instances": {
    "18411": 78,
    "18811": 13,
    "18211": 33,
    "6035": 2,
    "18401": 218
  },
  "library_sha256": "a46405b9917e756c344deb335089178201083976e0922fa56f329381ce023f5a",
  "warmup": 2,
  "repeat": 5,
  "samples": 3,
  "validation": "Every output compared,256-row chunks,inputs[-8,8],TF32 disabled,guards and256 sampled input values; full-range boundary validation separate"
}
```

Next derive transparent M/N/K/G rules, preserve runtime native/fallback guards, integrate into the public instance registry and validate the actual default API. Remaining fixed-target misses still require optimization.
