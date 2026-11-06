"""Symbolic MMAC accumulator exchange, including inactive rows and N tails."""
checks=0
initial=[[(lane%16,lane//16+v*4) for v in range(4)] for lane in range(64)]
stage1=[]
for lane,values in enumerate(initial):
    b0=(lane//16)&1
    r0,r1,r2,r3=values
    peer=initial[lane^16]
    pb0=((lane^16)//16)&1
    t0=peer[0] if pb0 else peer[1]
    t1=peer[2] if pb0 else peer[3]
    stage1.append([t0 if b0 else r0,r1 if b0 else t0,
                   t1 if b0 else r2,r3 if b0 else t1])
result=[]
for lane,values in enumerate(stage1):
    b1=(lane//16)&2
    r0,r1,r2,r3=values
    peer=stage1[lane^32]
    pb1=((lane^32)//16)&2
    t0=peer[0] if pb1 else peer[2]
    t1=peer[1] if pb1 else peer[3]
    got=[t0 if b1 else r0,t1 if b1 else r1,
         r2 if b1 else t0,r3 if b1 else t1]
    assert got==[(lane%16,(lane//16)*4+v) for v in range(4)]
    result.append(got);checks+=4

for waves in [2,4]:
 for nr in [1,2]:
  bn=waves*16*nr
  for m in [0,1,2,3,8,15,16,17,31,35]:
   for n in [1,4,17,31,32,33,60,64,128,260]:
    written=set()
    for mb in range(0,m,16):
     for nb in range(0,n,bn):
      for wave in range(waves):
       for ni in range(nr):
        for lane in range(64):
         for row,col in result[lane]:
          r,c=mb+row,nb+wave*16*nr+ni*16+col
          if r<m and c<n:
           assert (r,c) not in written
           written.add((r,c));checks+=1
    assert written=={(r,c) for r in range(m) for c in range(n)}
print('WAVE_STORE_C_MAPPING_PASS',checks)
