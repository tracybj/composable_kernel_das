// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t s){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspPriorityKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t s){switch(id){
case 14200:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,2,0>>(args,g,m,n,s);
case 14201:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,2,1>>(args,g,m,n,s);
case 14202:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,2,2>>(args,g,m,n,s);
case 14210:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,2,0>>(args,g,m,n,s);
case 14211:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,2,1>>(args,g,m,n,s);
case 14212:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,2,2>>(args,g,m,n,s);
case 14800:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,8,0>>(args,g,m,n,s);
case 14801:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,8,1>>(args,g,m,n,s);
case 14802:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,8,2>>(args,g,m,n,s);
case 14810:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,8,0>>(args,g,m,n,s);
case 14811:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,8,1>>(args,g,m,n,s);
case 14812:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,8,2>>(args,g,m,n,s);
case 15600:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,16,0>>(args,g,m,n,s);
case 15601:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,16,1>>(args,g,m,n,s);
case 15602:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<false,8,4,16,2>>(args,g,m,n,s);
case 15610:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,16,0>>(args,g,m,n,s);
case 15611:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,16,1>>(args,g,m,n,s);
case 15612:return run<ck_tile::GroupedInt8NtWaspPriorityProblem<true,8,4,16,2>>(args,g,m,n,s);
default:return baseline_launch(id,args,g,m,n,s);}}
