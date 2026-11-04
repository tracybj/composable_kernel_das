"""CPU proof of the packed16-bit wave transpose,not a GPU numeric gate."""
import random

def exchange(values):
    # One16-row MMAC block;quad h owns columns h+4*k before transpose.
    regs=[[(r[0]|r[1]<<16),(r[2]|r[3]<<16)] for r in values]
    old=[r[:] for r in regs]
    for h in range(4):
        for j in range(2):
            x,t=old[h][j],old[h^1][j]
            regs[h][j]=((t>>16)|(x&0xffff0000)) if h&1 else ((x&0xffff)|(t<<16)&0xffffffff)
    old=[r[:] for r in regs]
    for h in range(4):
        partner=h^2
        t=old[partner][0 if partner&2 else 1]
        regs[h]=[t,old[h][1]] if h&2 else [old[h][0],t]
    return [[r[0]&65535,r[0]>>16,r[1]&65535,r[1]>>16] for r in regs]

def main():
    rng=random.Random(20261104)
    tests=[[[h+4*k for k in range(4)] for h in range(4)]]
    for x in [0,1,32767,32768,65535,0x7f80,0x7fc0,0xffc0]:tests.append([[x]*4 for _ in range(4)])
    tests += [[[rng.randrange(65536) for _ in range(4)] for _ in range(4)] for _ in range(4096)]
    for values in tests:
        expected=[[values[col%4][col//4] for col in range(h*4,h*4+4)] for h in range(4)]
        assert exchange(values)==expected,(values,exchange(values),expected)
    print('PACKED_OUTPUT_LAYOUT_PASS',len(tests),'quad scenarios;wave rows independent')

if __name__=='__main__':main()
