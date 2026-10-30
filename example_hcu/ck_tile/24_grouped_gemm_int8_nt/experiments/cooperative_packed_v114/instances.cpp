// SPDX-License-Identifier: MIT
#include "../../cached_weights/cached_dispatch.cpp"
#include "cooperative/kernel.hpp"
namespace {
template<class P> int run_coop(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+255)/256,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtCooperativePackedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){
 if(id>=50001 && id<=50012)return ck_tile_int8_cached_launch_private(id-50000,0,args,g,m,n,stream);
 switch(id){
case 43001:return run_coop<ck_tile::GroupedInt8NtCooperativePackedProblem<false,16,4,2,true,true,0,false,true,0,16>>(args,g,m,n,stream);
case 43101:return run_coop<ck_tile::GroupedInt8NtCooperativePackedProblem<false,8,4,2,true,true,0,false,true,0,8>>(args,g,m,n,stream);
default:return ck_tile_int8_cached_legacy_dispatch_private(id,args,g,m,n,stream);}}
