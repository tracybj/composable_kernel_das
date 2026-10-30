// SPDX-License-Identifier: MIT
// Byte-tag diagnostic for independent producer/consumer layout derivation.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
namespace ck_tile {
struct Int8LdsMapProbeProblem { static constexpr int Threads=256,Rows=192,Slots=2,StageBytes=32768,SharedBytes=65536; };
struct Int8LdsMapProbePipeline {
 using Rsrc=int __attribute__((ext_vector_type(4)));
 CK_TILE_DEVICE static Rsrc descriptor(const void* a,int bytes){
  const auto p=reinterpret_cast<uintptr_t>(a);return Rsrc{int(p),int(p>>32),bytes,0x00020000};
 }
 CK_TILE_DEVICE static void run(const int* a,int* out,int* fragments,int mode,int valid,int pitch,unsigned char* smem){
  constexpr int Words=Int8LdsMapProbeProblem::SharedBytes/4;
  const int tid=get_thread_id(),lane=tid%64,wave=tid/64;
  for(int i=tid;i<Words;i+=256)reinterpret_cast<int*>(smem)[i]=0x6a6a6a6a;
  __syncthreads();const auto r=descriptor(a,valid*pitch);
  for(int stage=0;stage<2;++stage){
   #pragma unroll
   for(int i=0;i<6;++i){
    const int rg=lane/8,chunk=lane%8;
    const int row=mode==0?wave*48+i*8+rg:wave+4*rg+32*i;
    const int col=mode==0?(chunk^rg)*16:chunk*16;
    unsigned off=row*pitch+stage*128+col;if(row>=valid)off|=0x80000000u;
    const unsigned dest=__builtin_amdgcn_readfirstlane(stage*32768u+(mode==0?wave*6144u+i*1024u:wave*1024u+i*4096u)+(mode==2?(wave<<17):0u));
    asm volatile("s_mov_b32 m0, %1\n\tbuffer_load_dwordx4 %0, %2, 0 offen lds\n"
                 ::"v"(off),"s"(dest),"s"(r):"memory","m0");
   }
  }
  asm volatile("s_waitcnt vmcnt(0)\n\ts_mov_b32 m0, 0xffff\n":::"memory","m0");
  __syncthreads();
  for(int i=tid;i<Words;i+=256)out[i]=reinterpret_cast<int*>(smem)[i];
  // Observed matrix-fragment address function,independently expressed.
  const unsigned r0=lane%16,h=lane/16;
  const unsigned base=(r0%4)*1024+(r0/4)*128+h*16+(r0%4)*32;
  for(int stage=0;stage<2;++stage)for(int half=0;half<2;++half){
   #pragma unroll
   for(int mi=0;mi<12;++mi){
    const unsigned correction=(mi%2==0)?0:(half==0?((lane==47||lane==63)?1024:0):((r0>=14||lane==45||lane==61)?1024:0));
    const unsigned addr=static_cast<unsigned>(reinterpret_cast<uintptr_t>(smem))+stage*32768+base+(mi/2)*4096+(mi%2)*512+half*64-correction;
    using V=int __attribute__((ext_vector_type(4)));V v;
    asm volatile("ds_read_b128 %0, %1\n\ts_waitcnt lgkmcnt(0)" :"=v"(v):"v"(addr):"memory");
    if(wave==0){const int at=((stage*2+half)*12+mi)*64*4+lane*4;for(int j=0;j<4;++j)fragments[at+j]=v[j];}
   }
  }
 }
};
struct Int8LdsMapProbeKernel {
 CK_TILE_DEVICE void operator()(const int* a,int* out,int* fragments,int mode,int valid,int pitch) const {
  __shared__ __align__(16) unsigned char smem[Int8LdsMapProbeProblem::SharedBytes];
  Int8LdsMapProbePipeline::run(a,out,fragments,mode,valid,pitch,smem);
 }
};
}
extern "C" int int8_lds_map_probe(const int* a,int* out,int* fragments,int mode,int valid,int pitch){
 if(mode<0||mode>2||valid<1||valid>192||pitch<256||pitch%16)return -1;
 ck_tile::launch_kernel(ck_tile::stream_config{nullptr,false},ck_tile::make_kernel<256,1>(ck_tile::Int8LdsMapProbeKernel{},dim3(1),dim3(256),0,a,out,fragments,mode,valid,pitch));
 return hipGetLastError()==hipSuccess?0:-100;
}
