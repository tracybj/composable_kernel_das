// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+127)/128,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtNReuseKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 31001:return run<ck_tile::GroupedInt8NtNReuseProblem<false,1,0>>(args,g,m,n,s);
case 31004:return run<ck_tile::GroupedInt8NtNReuseProblem<false,4,0>>(args,g,m,n,s);
case 31011:return run<ck_tile::GroupedInt8NtNReuseProblem<false,1,1024>>(args,g,m,n,s);
case 31014:return run<ck_tile::GroupedInt8NtNReuseProblem<false,4,1024>>(args,g,m,n,s);
case 31101:return run<ck_tile::GroupedInt8NtNReuseProblem<true,1,0>>(args,g,m,n,s);
case 31104:return run<ck_tile::GroupedInt8NtNReuseProblem<true,4,0>>(args,g,m,n,s);
case 31111:return run<ck_tile::GroupedInt8NtNReuseProblem<true,1,1024>>(args,g,m,n,s);
case 31114:return run<ck_tile::GroupedInt8NtNReuseProblem<true,4,1024>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
