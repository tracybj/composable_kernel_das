"""Prove exact Raw-B vector-copy coverage and MMAC-read ownership per wave."""
checks=0
for bk in [128,256]:
 for waves in [2,4]:
  wavebytes=16*bk;lds=[None]*(waves*wavebytes)
  for wave in range(waves):
   touched=set()
   for lane in range(64):
    for i in range(wavebytes//1024):
     vec=lane+i*64;row=vec//(bk//16);ko=(vec%(bk//16))*16
     assert ko%16==0
     for byte in range(16):
      off=wave*wavebytes+row*bk+((ko+byte)^((row&7)*16))
      assert wave*wavebytes<=off<(wave+1)*wavebytes and off not in touched
      touched.add(off);lds[off]=(wave*16+row,ko+byte);checks+=1
   assert len(touched)==wavebytes
  assert all(x is not None for x in lds)
  for wave in range(waves):
   for lane in range(64):
    for kk in range(bk//32):
     row=lane%16;ko=kk*32+(lane//16)*8
     for byte in range(8):
      off=wave*wavebytes+row*bk+((ko+byte)^((row&7)*16))
      assert lds[off]==(wave*16+row,ko+byte);checks+=1
print('WAVE_PRIVATE_B_MAP_PASS',checks)
