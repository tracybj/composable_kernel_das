"""Bit-exact lane-map model for a future convert-before-shuffle epilogue.
This validates the 16-bit exchange only; it is not a GPU performance result.
"""
import random
rng=random.Random(20261105)
for trial in range(10000):
    original=[[rng.randrange(65536) for k in range(4)] for h in range(4)]
    v=[row[:] for row in original]
    exchange=[(r[0] if h&1 else r[1])|((r[2] if h&1 else r[3])<<16) for h,r in enumerate(v)]
    for h,r in enumerate(v):
        t=exchange[h^1];lo=t&65535;hi=t>>16
        if h&1:r[0]=lo;r[2]=hi
        else:r[1]=lo;r[3]=hi
    exchange=[(r[0] if h&2 else r[2])|((r[1] if h&2 else r[3])<<16) for h,r in enumerate(v)]
    for h,r in enumerate(v):
        t=exchange[h^2];lo=t&65535;hi=t>>16
        if h&2:r[0]=lo;r[1]=hi
        else:r[2]=lo;r[3]=hi
    assert v==[list(row) for row in zip(*original)]
print('PACKED_16BIT_LANE_TRANSPOSE_PASS',10000)
