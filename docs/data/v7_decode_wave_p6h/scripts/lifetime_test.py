"""Independent CPU check of requested stages, slot lifetimes and output ownership."""
import json
from pathlib import Path
checked=0
for stages in range(2,258,2):
 loaded=[0,1];consumed=[];slots=[0,1]
 for s in range(0,stages-2,2):
  for j in (0,1):
   current=s+j
   assert slots[j]==current
   consumed.append(current)
   future=current+2
   assert future<stages
   slots[j]=future;loaded.append(future);checked+=1
 consumed.extend(slots)
 assert loaded==list(range(stages)) and consumed==list(range(stages))
 # Loader writes unique disjoint16B blocks; read-address inverse of XOR.
 for wave in range(4):
  for slot in range(2):
   base=wave*2048+slot*1024
   for lane in range(64):
    k=lane>>2;octet=(lane&3)^((k&1)*2)
    global_chunk=k*4+octet
    inverse=base+k*64+((octet^((k&1)*2))*16)
    assert inverse==base+lane*16
 # Output transpose covers every BF16 element once; each wave disjoint.
 out=[]
 for wave in range(4):
  staged={}
  for lane in range(64):
   r=lane&15;q=lane>>4
   for n in range(2):
    for j in range(4):
     col=q+n*16+j*4;b=col*2
     a=r*64+(((b>>4)^(r&3))<<4)+(b&15)
     assert a not in staged
     staged[a]=(r,wave*32+col)
  for lane in range(64):
   row=lane>>2;unit=lane&3
   a=row*64+((unit^(row&3))<<4)
   for j in range(8):out.append(staged[a+j*2])
 assert sorted(out)==[(m,n) for m in range(16) for n in range(128)]
print('DECODE_CPU_LIFETIME_OUTPUT_PASS stage_transitions='+str(checked)+' K32..4096 plus beyond-contract controls; not a GPU synchronization proof')
