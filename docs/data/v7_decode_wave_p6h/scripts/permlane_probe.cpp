#include <hip/hip_runtime.h>
__device__ unsigned perm16(unsigned,unsigned,unsigned,unsigned,bool,bool) __asm("llvm.amdgcn.permlane16.i32");
extern "C" __global__ void probe(unsigned* p){unsigned v=p[threadIdx.x];p[threadIdx.x]=perm16(0,v,0x33221100u,0x77665544u,false,false);}
