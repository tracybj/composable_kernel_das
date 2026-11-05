"""CPU contract proof cases, not a replacement for the GPU full-range gate."""
import struct,math,random
rng=random.Random(20261105)
def f32(x):
 try:return struct.unpack('<f',struct.pack('<f',x))[0]
 except OverflowError:return math.copysign(float('inf'),x)
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(b):return struct.unpack('<f',struct.pack('<I',b))[0]
def finite(f):return bits(f)&0x7fffffff<0x7f800000
def generic(acc,f):
 r=f32(f32(float(acc))*f);b=bits(r)
 return ((b>>16)|0x40) if (b&0x7fffffff)>0x7f800000 else ((b+0x7fff+((b>>16)&1))&0xffffffff)>>16
def fast(acc,f):
 b=bits(f32(f32(float(acc))*f))
 return ((b+0x7fff+((b>>16)&1))&0xffffffff)>>16
edge=[0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001]
checks=0
for acc in [0,1,-1,2,-2,2147483647,-2147483648,16777217,-16777217]:
 for b in edge:
  f=value(b)
  if finite(f):assert not math.isnan(f32(f32(float(acc))*f));assert fast(acc,f)==generic(acc,f);checks+=1
for _ in range(100000):
 acc=rng.randrange(-2147483648,2147483648);f=value(rng.getrandbits(32))
 if finite(f):assert not math.isnan(f32(f32(float(acc))*f));assert fast(acc,f)==generic(acc,f);checks+=1
# Finite input scales can overflow their FP32 product: the branch must reject it.
assert math.isfinite(value(0x7f7fffff)) and not finite(f32(value(0x7f7fffff)*2.0))
# Zero*Inf must continue through the generic NaN-capable converter.
assert not finite(float('inf')) and math.isnan(f32(0.0*float('inf')))
print('FINITE_FACTOR_CPU_PASS',checks)
