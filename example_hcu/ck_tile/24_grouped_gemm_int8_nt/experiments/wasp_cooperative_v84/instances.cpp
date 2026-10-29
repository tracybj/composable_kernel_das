// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+511)/512,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCoopKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 26001:return run<ck_tile::GroupedInt8NtWaspCoopProblem<false,1,0>>(args,g,m,n,s);
case 26002:return run<ck_tile::GroupedInt8NtWaspCoopProblem<false,2,0>>(args,g,m,n,s);
case 26011:return run<ck_tile::GroupedInt8NtWaspCoopProblem<false,1,1024>>(args,g,m,n,s);
case 26012:return run<ck_tile::GroupedInt8NtWaspCoopProblem<false,2,1024>>(args,g,m,n,s);
case 26101:return run<ck_tile::GroupedInt8NtWaspCoopProblem<true,1,0>>(args,g,m,n,s);
case 26102:return run<ck_tile::GroupedInt8NtWaspCoopProblem<true,2,0>>(args,g,m,n,s);
case 26111:return run<ck_tile::GroupedInt8NtWaspCoopProblem<true,1,1024>>(args,g,m,n,s);
case 26112:return run<ck_tile::GroupedInt8NtWaspCoopProblem<true,2,1024>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
