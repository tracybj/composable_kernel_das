"""Probe actual VMEM-to-LDS bytes and observed fragment addresses on physicalGPU7."""
from pathlib import Path
import ctypes as C,json,hashlib,os,collections,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
import torch
torch.set_num_threads(4);assert all(os.environ.get(x)=='7' for x in ['HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','DTK_VISIBLE_DEVICES'])
e=Path(__file__).resolve().parents[2];b=e/'build/v118';out=e/'results/lds_map_v118_verified';out.mkdir(exist_ok=True)
hip=C.CDLL('/opt/dtk/lib/libgalaxyhip.so.5');pci=C.create_string_buffer(64);assert hip.hipDeviceGetPCIBusId(pci,64,0)==0 and pci.value.decode().lower()=='0000:f5:00.0'
lib=C.CDLL(str(b/'libmap.so'));fn=lib.int8_lds_map_probe;fn.argtypes=[C.c_void_p]*3+[C.c_int]*3;fn.restype=C.c_int;records=[]
sentinel=0x6a6a6a6a
for valid,pitch in [(192,256),(128,256),(32,256),(192,272)]:
 a=(torch.arange(valid*pitch//4,device='cuda',dtype=torch.int32)+1).reshape(valid,pitch//4);olda=a.clone()
 for mode in [0,1,2]:
  raw=torch.full((16384+32,),-17,device='cuda',dtype=torch.int32);dump=raw[16:-16];fr=torch.full((12288+32,),-23,device='cuda',dtype=torch.int32);frag=fr[16:-16]
  assert fn(a.data_ptr(),dump.data_ptr(),frag.data_ptr(),mode,valid,pitch)==0;torch.cuda.synchronize();cpu=dump.cpu().tolist();fcpu=frag.cpu().tolist()
  expected=[]
  for stage in range(2):
   for row in range(valid):
    expected.extend(row*(pitch//4)+stage*32+j+1 for j in range(32))
  got=[v for v in cpu if v not in [sentinel,0]]
  assert collections.Counter(got)==collections.Counter(expected),(valid,pitch,mode,'payload',len(got))
  assert bool((raw[:16]==-17).all()&(raw[-16:]==-17).all()&(fr[:16]==-23).all()&(fr[-16:]==-23).all()) and torch.equal(a,olda)
  if mode in [0,1,2]:
   model=[sentinel]*16384
   for stage in range(2):
    for w in range(4):
     for lane in range(64):
      for i in range(6):
       rg=lane//8;chunk=lane%8;row=w*48+i*8+rg if mode==0 else w+4*rg+32*i;col=(chunk^rg)*16 if mode==0 else chunk*16;dest=stage*32768+(w*6144+i*1024+lane*16 if mode==0 else w*1024+i*4096+((lane*16+32*w)%1024 if mode==2 else lane*16))
       for j in range(4):model[dest//4+j]=row*(pitch//4)+stage*32+col//4+j+1 if row<valid else 0
   assert cpu==model,(valid,pitch,mode,'independent exact dword model')
  if mode==2:
   # Derive a direct address from logical row/chunk without vendor lane masks.
   for stage in range(2):
    for row in range(192):
     for chunk in range(8):
      w=row%4;rg=(row%32)//4;panel=row//32;at=stage*8192+panel*1024+w*256+((rg*128+chunk*16+32*w)%1024)//4
      for j in range(4):assert cpu[at+j]==(row*(pitch//4)+stage*32+chunk*4+j+1 if row<valid else 0)
  bad=0;first=[]
  for stage in range(2):
   for half in range(2):
    for mi in range(12):
     for lane in range(64):
      row=mi*16+lane%16
      for j in range(4):
       at=((stage*2+half)*12+mi)*256+lane*4+j;want=row*(pitch//4)+stage*32+half*16+(lane//16)*4+j+1 if row<valid else 0
       if fcpu[at]!=want:
        bad+=1
        if len(first)<12:first.append(dict(stage=stage,half=half,mi=mi,lane=lane,word=j,got=fcpu[at],want=want))
  if mode==2:assert bad==0,(valid,pitch,bad,first)
  name=f'm{mode}_rows{valid}_pitch{pitch}';(out/(name+'.json')).write_text(json.dumps(dict(dump=cpu,fragments=fcpu),separators=(',',':')))
  record=dict(mode=mode,valid_rows=valid,pitch=pitch,payload_multiset=True,guards=True,inputs_unchanged=True,fragment_expected_linear_bad=bad,first_bad=first,library_sha256=hashlib.sha256((b/'libmap.so').read_bytes()).hexdigest());records.append(record);print(json.dumps(record),flush=True)
report=dict(status='PASS',physical_gpu=7,pci=pci.value.decode(),exact_dword_model_all_modes=True,mode2_logical_fragment_bad=0,records=records);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('V118_MAP_DONE',len(records))
