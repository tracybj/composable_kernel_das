// SPDX-License-Identifier: MIT
#include "../../cached_weights/cached_dispatch.cpp"
#include "m128_n192/kernel.hpp"
namespace {
template<class P,bool SmallM> int run_occ(const void* args,int g,int m,int n,hipStream_t stream){
 constexpr int BM=SmallM?128:256;const int mt=(m+BM-1)/BM,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 if constexpr(SmallM)ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspOccupancyM128Kernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 else ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspPublicWideNKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){
 if(id>=50001 && id<=50012)return ck_tile_int8_cached_launch_private(id-50000,0,args,g,m,n,stream);
 switch(id){
case 42001:return run_occ<ck_tile::GroupedInt8NtWaspOccupancyM128Problem<false,6,1,4,true,true,0,false,true,0,6>,true>(args,g,m,n,stream);
case 42002:return run_occ<ck_tile::GroupedInt8NtWaspOccupancyM128Problem<false,6,1,2,true,true,0,false,true,0,6>,true>(args,g,m,n,stream);
case 42102:return run_occ<ck_tile::GroupedInt8NtWaspPublicWideNProblem<false,10,4,2,true,true,0,false,true,0,10>,false>(args,g,m,n,stream);
case 42104:return run_occ<ck_tile::GroupedInt8NtWaspPublicWideNProblem<false,10,4,4,false,true,0,false,true,0,10>,false>(args,g,m,n,stream);
case 42204:return run_occ<ck_tile::GroupedInt8NtWaspPublicWideNProblem<false,12,4,4,false,true,0,false,true,0,12>,false>(args,g,m,n,stream);
default:return ck_tile_int8_cached_legacy_dispatch_private(id,args,g,m,n,stream);}}
