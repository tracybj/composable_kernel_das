// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspOneBarrierKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 18200:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,2,true,false>>(args,g,m,n,s);
case 18201:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,2,true,true>>(args,g,m,n,s);
case 18210:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,2,true,false>>(args,g,m,n,s);
case 18211:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,2,true,true>>(args,g,m,n,s);
case 18400:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,4,true,false>>(args,g,m,n,s);
case 18401:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,4,true,true>>(args,g,m,n,s);
case 18410:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,4,true,false>>(args,g,m,n,s);
case 18411:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,4,true,true>>(args,g,m,n,s);
case 18800:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,8,true,false>>(args,g,m,n,s);
case 18801:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<false,8,4,8,true,true>>(args,g,m,n,s);
case 18810:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,8,true,false>>(args,g,m,n,s);
case 18811:return run<ck_tile::GroupedInt8NtWaspOneBarrierProblem<true,8,4,8,true,true>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
