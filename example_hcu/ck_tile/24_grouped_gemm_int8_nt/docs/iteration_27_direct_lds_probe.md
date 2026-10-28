# Iteration 27: native direct-to-LDS layout probe

Physical GPU7/gfx936; one 64-thread CTA, 16 bytes per lane, 256 uint32 input values. This is a copy/layout probe, not GEMM or a performance result.

Using CK hcu_async_buffer_load_asm (dwordx4), s_waitcnt vmcnt(0), then a block LDS barrier:

- Linear lane-vector load reconstructs the input exactly.
- Per-row XOR of the four K vectors, followed by inverse XOR reads, reconstructs input exactly.
- Invalid odd-lane accesses with buffer offsets equal to SRD range write zeros into LDS; prefilled shared-memory sentinels are replaced. Output guards remain intact.

These observations support the planned native input layout and masking. Full GEMM correctness, synchronization across producer/consumer waves, larger sizes and performance must still be tested.

Copy library SHA256: 4fd75b1352624f63f21ea6024a1335f356721a4ceb7c48b5ade5b7f0ebeb82e8

Source: experiments/direct_lds_v27/copy_probe.cpp and check_copy.py.

```json
[
  {
    "mode": 0,
    "equal": true,
    "guard": true,
    "head": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ]
  },
  {
    "mode": 1,
    "equal": true,
    "guard": true,
    "head": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      23,
      24,
      25,
      26,
      27,
      28,
      29,
      30,
      31
    ]
  },
  {
    "mode": 2,
    "equal": true,
    "guard": true,
    "head": [
      0,
      1,
      2,
      3,
      0,
      0,
      0,
      0,
      8,
      9,
      10,
      11,
      0,
      0,
      0,
      0,
      16,
      17,
      18,
      19,
      0,
      0,
      0,
      0,
      24,
      25,
      26,
      27,
      0,
      0,
      0,
      0
    ]
  }
]
```
