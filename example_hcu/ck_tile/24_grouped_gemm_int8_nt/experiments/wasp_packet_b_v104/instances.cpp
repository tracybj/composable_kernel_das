// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../../cached_weights/kernel.hpp"
#include "kernel.hpp"
namespace {
template<class P,bool Packet> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+255)/256,nt=(n+255)/256;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 if constexpr(Packet)ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspPacketKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 else ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCachedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){switch(id){
case 32001:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,true,0>,false>(args,g,m,n,stream);
case 37001:return run<ck_tile::GroupedInt8NtWaspPacketProblem<false,8,4,4,true,true,0,false,true,0>,true>(args,g,m,n,stream);
case 37002:return run<ck_tile::GroupedInt8NtWaspPacketProblem<false,8,4,2,true,true,0,false,true,0>,true>(args,g,m,n,stream);
case 37008:return run<ck_tile::GroupedInt8NtWaspPacketProblem<false,8,4,8,true,true,0,false,true,0>,true>(args,g,m,n,stream);
default:return baseline_launch(id,args,g,m,n,stream);}}
