// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../../cached_weights/kernel.hpp"
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+255)/256,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspWideNKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
int control(const void* args,int g,int m,int n,hipStream_t stream){using P=ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,true,0>;const int mt=(m+255)/256,nt=(n+255)/256;const dim3 grid(((mt+3)/4)*4*nt,1,g);ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCachedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){switch(id){
case 32001:return control(args,g,m,n,stream);
case 39001:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,12,4,4,true,true,0,false,true,0,12>>(args,g,m,n,stream);
case 39002:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,12,4,2,true,true,0,false,true,0,12>>(args,g,m,n,stream);
case 39008:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,12,8,4,true,true,0,false,true,0,12>>(args,g,m,n,stream);
case 39101:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,8,4,4,true,true,0,false,true,0,8>>(args,g,m,n,stream);
case 39108:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,8,8,4,true,true,0,false,true,0,8>>(args,g,m,n,stream);
case 39116:return run<ck_tile::GroupedInt8NtWaspWideNProblem<false,8,16,4,true,true,0,false,true,0,8>>(args,g,m,n,stream);
default:return baseline_launch(id,args,g,m,n,stream);}}
