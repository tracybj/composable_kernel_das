// SPDX-License-Identifier: MIT
#include "problem.hpp"
#include <hip/hip_runtime.h>
using P=ck_tile::GroupedInt8NtWaspNativeProblem<false,8,4>;
using Pipe=P::Pipeline;
namespace d=ck_tile::wasp_int8_native_detail;
template<int MODE> __global__ void diag(const int8_t* a,const int8_t* b,int32_t* c,int sa,int sb)
{
 using namespace ck_tile;
 __shared__ __align__(16) int8_t smem[65536];
 const int tid=threadIdx.x,w=__builtin_amdgcn_readfirstlane(tid/64),l=tid%64;
 if constexpr(MODE==0) {
  for(int stage=0;stage<2;++stage) {
   if(w<4){auto src=Pipe{}.make_a_prod_src(a,sa,256,w);Pipe{}.producer_fill(src,stage,w);d::wasp_wait_vmcnt0();}
   __syncthreads();
   for(int i=tid;i<256*128;i+=768){int row=i/128,k=i%128;int offset=row*128+(k^((row&7)*16));c[row*256+stage*128+k]=smem[stage*32768+offset];}
   __syncthreads();
  }
 } else if constexpr(MODE==1) {
  if(w<8){
   auto r=d::wasp_make_buffer_rsrc(b,sb,256*sb,true);
   unsigned row=l&15,h=l>>4,nrow=w*32+row;
   Pipe::BPairs pairs{(static_cast<uint64_t>(h*16)<<32)|nrow,(static_cast<uint64_t>(h*16)<<32)|(nrow+16)};
   static_for<0,2,1>{}([&](auto ss){constexpr int stage=decltype(ss)::value;Pipe::BBuf buf;d::WaspVmemDiag vd{};Pipe{}.load_B_stage(r,pairs,stage,buf,vd);d::wasp_wait_vmcnt0();
    static_for<0,2,1>{}([&](auto hh){constexpr int half=decltype(hh)::value;
     static_for<0,2,1>{}([&](auto nn){constexpr int ni=decltype(nn)::value;
      static_for<0,16,1>{}([&](auto jj){constexpr int j=decltype(jj)::value;
       int32_t word=buf[half][ni].d4[j/4];int32_t value=static_cast<int8_t>(static_cast<uint32_t>(word)>>((j%4)*8));
       c[(nrow+ni*16)*256+stage*128+half*64+h*16+j]=value;
      });
     });
    });
   });
  }
 } else if constexpr(MODE==2) {
  Pipe::AccTile acc;Pipe{}.run_raw_into(a,sa,256,b,sb,256,0,2,acc,smem);
  if(w>=4){for(int mi=0;mi<16;++mi)for(int ni=0;ni<2;++ni)for(int vi=0;vi<4;++vi)c[(mi*16+l%16)*256+(w-4)*32+ni*16+l/16+vi*4]=acc.v[mi*2+ni][vi];}
 } else {c[tid]=get_warp_id()*64+get_lane_id();}
}
extern "C" int diag_run(const int8_t* a,const int8_t* b,int32_t* c,int sa,int sb,int mode){
 if(mode==0)hipLaunchKernelGGL(diag<0>,dim3(1),dim3(768),0,0,a,b,c,sa,sb);
 else if(mode==1)hipLaunchKernelGGL(diag<1>,dim3(1),dim3(768),0,0,a,b,c,sa,sb);
 else if(mode==2)hipLaunchKernelGGL(diag<2>,dim3(1),dim3(768),0,0,a,b,c,sa,sb);
 else hipLaunchKernelGGL(diag<3>,dim3(1),dim3(768),0,0,a,b,c,sa,sb);
 return static_cast<int>(hipDeviceSynchronize());
}
