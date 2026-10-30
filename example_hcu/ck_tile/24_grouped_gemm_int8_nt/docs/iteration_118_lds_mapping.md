# Iteration118:GPU byte proof of producer/consumer LDS mapping

Diagnostic only,no production or timing claim. Built independent CK Problem/Pipeline/Kernel around a256-thread,4wave stage probe. Compare(0)existing XOR slab distribution,(1)interleaved source rows withoutm0 highbits,(2)the same rows withwave<<17 in m0. Initialize64KiB LDS witha sentinel,copy2stages ofraw row/K-identifiable INT32tags through16B VMEM-to-LDS,readbackthe complete LDS,andcaptureall12MIx2Khalvesx64lanesx4dword fragments perstage. Input isonlya byte-identifiable diagnostic,notnew GEMM arithmetic.

12profilesPASS:3modesx(valid rows192/128/32 withpitch256,androws192/pitch272). Eachobserved dword inall64KiB matches anindependent CPU model;allsource payload tags appear exactlyonce;output guards andcomplete inputs unchanged. For mode2,all12288fragment dwords perprofile exactlymatch logicalrow=MI*16+lane%16 andK=stage*128+half*64+(lane/16)*16+word*4. Mode0/1 pairedwiththemode2consumer addresses are deliberate negative layout controls,not numericalGEMM failures.

Derived map,forproducer wave w=0..3,lane l=0..63,vector i=0..5:

```
source_row = w + 4*(l/8) + 32*i
source_Kbyte = stage*128 + 16*(l%8)
LDS_byte = stage*32768 + i*4096 + w*1024 + ((16*l + 32*w) % 1024)
```

Thus m0'swave<<17 tag appliesa32*w-byte rotation inside each1KiB wave-copy region inthisverified gfx936/DTK mode. Do notgeneralize toother architectures/instructions. Thecopy sourceperwave interleavesrowsinstead ofour previouscontiguous48-row slabs. Payload remains24KiB per192x128stage insidea32KiB slot.

Anindependent consumer address avoids vendorlane-case masks. Forlogicalrow r,source16Bchunk c=0..7:

```
w=r%4; rg=(r%32)/4; panel=r/32
LDS_byte = stage*32768 + panel*4096 + w*1024
         + ((128*rg + 16*c + 32*w) % 1024)
```

ForMI=2q/parity andlane r0=lane%16,common=128*(r0/4)+16*(lane/16)+32*(r0%4). EvenMI usescommon(+64forhighKhalf);oddMI wrapscommon+512(+64high) within1KiB. Addq*4096andwavebase1024*(r0%4). CompleteGPUreadbackproves these addresses,including thewrapping lanes. Initialtranscription missed twohigh-half/oddMI boundary lanes45/61;itproduced96badfragmentdwords ona full192-rowprofilewhilepayloadwascorrect. Corrected via theindependent map,andkept originalsummary/source/binary underresults/lds_map_v118 andbuild/v118/initial. The final mode2has0bad fragmentsinall4profiles.

Firstcompiler attemptallocatedaVGPR tom0'ssource. Explicitreadfirstlane onthewave-uniform destination producesa scalaroperand. Thiswasacompilation issue,notGPUavailability. FinalbinarySHA andprofiledetails areintheJSON. No rawvendorinstructionblock iscommitted;formulasandprobeareowncode.

Next integrate the proven interleaved/rotated publication andfragmentbases intoM192transposed GEMM,thenvalidate4+4staggered consumers andpriority1 independently. PriorM192prototype'snegative result didnotinclude thismapping. RequirefullrangeINT8,tail/odd/evenstagechecks andsame-moduletiming beforepromotion. Formalv112remains1178numericPASS/1170targets/8gaps;FP16/BF16interfacesunchanged.
