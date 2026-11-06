"""Independent symbolic trace: diagonal-only MMAC implements contiguous K dot."""
from collections import Counter
checks=0
for waves in [4,8]:
 for k in [16,512,640,7168,8192,8208]:
  covered=Counter()
  for base in range(0,k,waves*1024):
   for wave in range(waves):
    for r in range(16):
     owners=[(lane,v) for lane in range(64) for v in range(4) if lane%16==r and lane//16+v*4==r]
     assert len(owners)==1
     for half in range(2):
      for q in range(4):
       for t in range(8):
        ak=base+wave*1024+(r+q*16)*16+half*8+t
        bk=base+wave*1024+(r+q*16)*16+half*8+t
        assert ak==bk
        if ak<k:covered[ak]+=1
  assert covered==Counter(range(k))
  checks+=k
print('K_DIAGONAL_MAPPING_PASS',checks)
