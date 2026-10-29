// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspScheduleKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 28000:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,0,4,4,true,true,0>>(args,g,m,n,s);
case 28004:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,4,4,4,true,true,0>>(args,g,m,n,s);
case 28008:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,8,4,4,true,true,0>>(args,g,m,n,s);
case 28010:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,0,4,4,true,true,1024>>(args,g,m,n,s);
case 28014:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,4,4,4,true,true,1024>>(args,g,m,n,s);
case 28018:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<false,8,4,4,true,true,1024>>(args,g,m,n,s);
case 28100:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,0,4,4,true,true,0>>(args,g,m,n,s);
case 28104:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,4,4,4,true,true,0>>(args,g,m,n,s);
case 28108:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,8,4,4,true,true,0>>(args,g,m,n,s);
case 28110:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,0,4,4,true,true,1024>>(args,g,m,n,s);
case 28114:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,4,4,4,true,true,1024>>(args,g,m,n,s);
case 28118:return run<ck_tile::GroupedInt8NtWaspScheduleProblem<true,8,4,4,true,true,1024>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
