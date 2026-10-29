// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspNativeOnlyKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 35000:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<false,0,0>>(args,g,m,n,s);
case 35001:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<false,0,1>>(args,g,m,n,s);
case 35010:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<false,1024,0>>(args,g,m,n,s);
case 35011:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<false,1024,1>>(args,g,m,n,s);
case 35100:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<true,0,0>>(args,g,m,n,s);
case 35101:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<true,0,1>>(args,g,m,n,s);
case 35110:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<true,1024,0>>(args,g,m,n,s);
case 35111:return run<ck_tile::GroupedInt8NtWaspNativeOnlyProblem<true,1024,1>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
