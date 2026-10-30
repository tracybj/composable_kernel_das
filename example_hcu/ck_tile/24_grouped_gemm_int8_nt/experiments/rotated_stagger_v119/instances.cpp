// SPDX-License-Identifier: MIT
#include "../../cached_weights/cached_dispatch.cpp"
#include "../blas_geometry_v117/m192/kernel.hpp"
#include "m192/kernel.hpp"
namespace {
template<class P,bool Rotated> int run_m192(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(n+191)/192,nt=(m+255)/256;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 if constexpr(Rotated)ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspM192RotatedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 else ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspM192TransposeKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){
 if(id>=50001 && id<=50012)return ck_tile_int8_cached_launch_private(id-50000,0,args,g,m,n,stream);
 switch(id){
case 46001:return run_m192<ck_tile::GroupedInt8NtWaspM192RotatedProblem<true,8,4,12,false,true,1024,false,false,0,8,0>,true>(args,g,m,n,stream);
case 46002:return run_m192<ck_tile::GroupedInt8NtWaspM192RotatedProblem<true,8,4,12,false,true,1024,false,false,0,8,1>,true>(args,g,m,n,stream);
case 46003:return run_m192<ck_tile::GroupedInt8NtWaspM192RotatedProblem<true,4,4,12,false,false,1024,false,false,0,8,0>,true>(args,g,m,n,stream);
case 46004:return run_m192<ck_tile::GroupedInt8NtWaspM192RotatedProblem<true,4,4,12,false,false,1024,false,false,0,8,1>,true>(args,g,m,n,stream);
 case 45002:return run_m192<ck_tile::GroupedInt8NtWaspM192TransposeProblem<true,8,4,12,false,true,1024,false,false,0,8>,false>(args,g,m,n,stream);
 case 45003:return run<ck_tile::GroupedInt8NtWaspCachedProblem<true,4,4,16,false,false,1024,false,false,0>>(args,g,m,n,stream);
 default:return ck_tile_int8_cached_legacy_dispatch_private(id,args,g,m,n,stream);}}
