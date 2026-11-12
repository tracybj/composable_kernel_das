"""Independent address inverse and LDS slot/version safety checks, CPU only."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def run():
 address_checks=0;schedules=0
 for rows in [1,2,4]:
  for bk in [16,32]:
   mg=2 if rows==4 else 1;ng=4//mg;wm=rows//mg*16;wn=128//ng
   a={};b={}
   for row in range(((rows*16+512//bk-1)//(512//bk))*(512//bk)):
    for chunk in range(bk//8):
     physical=row*bk*2+(chunk^((row>>2)&(bk//8-1)))*16
     assert physical not in a;a[physical]=(row,chunk*8)
   for row in range(bk):
    for chunk in range(16):
     physical=(row//16)*4096+(row%16)*256+(chunk^((row&1)*4))*16
     assert physical not in b;b[physical]=(row,chunk*8)
   for wave in range(4):
    for lane in range(64):
     for h in range(bk//16):
      for m in range(rows//mg):
       row=(wave//ng)*wm+m*16+(lane&15);logical=h*32+(lane>>4)*8
       physical=row*bk*2+(logical^(((row>>2)&(bk//8-1))*16))
       # A reads half an aligned load chunk.
       assert a[physical//16*16]==(row,(logical//16)*8)
       address_checks+=1
      for r in range(wn//32):
       row=lane>>2;chunk=(wave%ng)*(wn//8)+r*4+(lane&3)
       physical=h*4096+row*256+(chunk^((row&1)*4))*16
       assert b[physical]==(h*16+row,chunk*8)
       address_checks+=1
 for depth in [4,8]:
  for stages in range(max(8,depth+1),513):
   slots={i:i for i in range(depth)};banks={0:0,1:1};readers=set()
   # Initial two readers have retired before the first overwrite.
   slots[0]=depth
   base=0;cycle=3*depth;consumed=[]
   def step(t):
    assert banks[t%3]==t,(depth,stages,t,banks)
    # All previous LDS reads retired at the boundary.
    readers.clear()
    if t+2<stages:
     assert slots[(t+2)%depth]==t+2,(depth,stages,t,slots)
     banks[(t+2)%3]=t+2;readers.add((t+2)%depth)
    if t+depth+1<stages:
     slot=(t+1)%depth
     assert slot not in readers,(depth,stages,t)
     slots[slot]=t+depth+1
    consumed.append(banks[t%3])
   while base+cycle+depth<stages:
    for j in range(cycle):step(base+j)
    base+=cycle
   for j in range(cycle+depth+1):
    if base+j<stages:step(base+j)
   assert consumed==list(range(stages));schedules+=1
 result=dict(ok=True,address_checks=address_checks,slot_schedules=schedules,cpu_only=True,not_gpu_precision=True)
 (ROOT/'h3_cpu_model_result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':run()
