// SPDX-License-Identifier: MIT
// The existing INT8 provider is compiled under a PRIVATE symbol. No old ABI
// owner or FP16/BF16 provider is defined by this extension.
#define ck_tile_grouped_int8_nt_launch ck_tile_int8_cached_legacy_dispatch_private
#include "../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
#include "m128/kernel.hpp"
#include "m64/kernel.hpp"
#include "m32/kernel.hpp"
#include "wide_n/kernel.hpp"
#include "pack_kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n,mt=(mm+255)/256,nt=(nn+255)/256;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(
  ck_tile::GroupedInt8NtWaspCachedKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
template<class P> int run_m128(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+127)/128,nt=(n+P::BN-1)/P::BN;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(
  ck_tile::GroupedInt8NtWaspSplitMKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}

template<class P> int run_m64(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+63)/64,nt=(n+P::BN-1)/P::BN;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(
  ck_tile::GroupedInt8NtWaspSmallMKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}

template<class P> int run_m32(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+31)/32,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspM32Kernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
template<class P> int run_m256_wide(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+255)/256,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspPublicWideNKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}

}
extern "C" int ck_tile_int8_cached_launch_private(unsigned route,unsigned legacy,const void* args,int g,int m,int n,hipStream_t stream){
 switch(route){
 case 0:return ck_tile_int8_cached_legacy_dispatch_private(legacy,args,g,m,n,stream);
 case 1:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,false,1>>(args,g,m,n,stream);
 case 2:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,1024,false,false,1>>(args,g,m,n,stream);
 case 3:return run<ck_tile::GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,0,false,false,1>>(args,g,m,n,stream);
 case 4:return run<ck_tile::GroupedInt8NtWaspCachedProblem<true,8,4,4,true,true,1024,false,false,1>>(args,g,m,n,stream);
 case 5:return run<ck_tile::GroupedInt8NtWaspCachedProblem<false,8,4,4,true,true,0,false,true,0>>(args,g,m,n,stream);
 case 6:return run_m128<ck_tile::GroupedInt8NtWaspSplitMProblem<false,8,4,2,true,true,0,false,true,0,8>>(args,g,m,n,stream);
 case 7:return run_m128<ck_tile::GroupedInt8NtWaspSplitMProblem<false,4,1,4,true,true,0,false,true,0,4>>(args,g,m,n,stream);
 case 8:return run_m64<ck_tile::GroupedInt8NtWaspSmallMProblem<false,8,1,2,true,true,0,false,true,1,8>>(args,g,m,n,stream);
 case 9:return run_m64<ck_tile::GroupedInt8NtWaspSmallMProblem<false,2,1,2,true,true,0,false,true,1,2>>(args,g,m,n,stream);
 case 10:return run_m32<ck_tile::GroupedInt8NtWaspM32Problem<false,8,1,2,true,true,0,false,true,1,8>>(args,g,m,n,stream);
 case 11:return run_m32<ck_tile::GroupedInt8NtWaspM32Problem<false,2,1,1,true,true,0,false,true,1,2>>(args,g,m,n,stream);
 case 12:return run_m256_wide<ck_tile::GroupedInt8NtWaspPublicWideNProblem<false,12,4,2,true,true,0,false,true,0,12>>(args,g,m,n,stream);
 default:return -1;
 }
}
extern "C" int ck_tile_int8_cached_pack_private(const ck_tile::Int8CachedWeightPackArgs* args,int g,int n,int k,hipStream_t stream){
 using P=ck_tile::Int8CachedWeightPackProblem;
 const dim3 grid((n+63)/64,k/64,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(
  ck_tile::Int8CachedWeightPackKernel<P>{},grid,dim3(P::Threads),0,args));
 return hipGetLastError()==hipSuccess?0:-100;
}
