// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../../grouped_int8_nt_wasp_kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 21010:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,1,4,true,true>>(args,g,m,n,s);
case 21011:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,1,4,true,true>>(args,g,m,n,s);
case 21020:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,2,4,true,true>>(args,g,m,n,s);
case 21021:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,2,4,true,true>>(args,g,m,n,s);
case 21040:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,4,4,true,true>>(args,g,m,n,s);
case 21041:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,4,4,true,true>>(args,g,m,n,s);
case 21050:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,5,4,true,true>>(args,g,m,n,s);
case 21051:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,5,4,true,true>>(args,g,m,n,s);
case 21080:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,8,4,true,true>>(args,g,m,n,s);
case 21081:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,8,4,true,true>>(args,g,m,n,s);
case 21160:return run<ck_tile::GroupedInt8NtWaspProblem<false,8,16,4,true,true>>(args,g,m,n,s);
case 21161:return run<ck_tile::GroupedInt8NtWaspProblem<true,8,16,4,true,true>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
