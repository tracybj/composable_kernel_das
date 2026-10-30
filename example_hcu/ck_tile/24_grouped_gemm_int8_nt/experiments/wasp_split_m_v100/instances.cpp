// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../../cached_weights/kernel.hpp"
#include "kernel.hpp"
namespace {
template<class P,bool Small> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,bm=Small?128:256,bn=P::BN,mt=(mm+bm-1)/bm,nt=(nn+bn-1)/bn;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 if constexpr(Small)ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspSplitMKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 else ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCachedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){switch(id){
case 34001:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,false,1>,false>(args,g,m,n,stream);
case 34011:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,1024,false,false,1>,false>(args,g,m,n,stream);
case 34101:return run<ck_tile::GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,0,false,false,1>,false>(args,g,m,n,stream);
case 34111:return run<ck_tile::GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,1024,false,false,1>,false>(args,g,m,n,stream);
case 32001:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,true,0>,false>(args,g,m,n,stream);
case 36001:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,4,true,true,0,false,false,1>,true>(args,g,m,n,stream);
case 36011:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,4,true,true,1024,false,false,1>,true>(args,g,m,n,stream);
case 36101:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<true,8,4,4,true,true,0,false,false,1>,true>(args,g,m,n,stream);
case 36111:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<true,8,4,4,true,true,1024,false,false,1>,true>(args,g,m,n,stream);
case 36201:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,4,true,true,0,false,true,0>,true>(args,g,m,n,stream);
case 36202:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,2,true,true,0,false,true,0>,true>(args,g,m,n,stream);
case 36208:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,8,true,true,0,false,true,0>,true>(args,g,m,n,stream);
case 36301:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,2,1,4,true,true,0,false,false,1,2>,true>(args,g,m,n,stream);
case 36302:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,2,1,4,true,true,0,false,true,0,2>,true>(args,g,m,n,stream);
case 36401:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,4,1,4,true,true,0,false,false,1,4>,true>(args,g,m,n,stream);
case 36402:return run<ck_tile::GroupedInt8NtWaspSplitMProblem<false,4,1,4,true,true,0,false,true,0,4>,true>(args,g,m,n,stream);
default:return baseline_launch(id,args,g,m,n,stream);}}
