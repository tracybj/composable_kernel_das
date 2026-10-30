// SPDX-License-Identifier: MIT
// The existing INT8 provider is compiled under a PRIVATE symbol. No old ABI
// owner or FP16/BF16 provider is defined by this extension.
#define ck_tile_grouped_int8_nt_launch ck_tile_int8_cached_legacy_dispatch_private
#include "../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "core_kernel.hpp"
#include "instances.hpp"
#include "pack_kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mm=P::Transpose?n:m, nn=P::Transpose?m:n;
 const int mt=(mm+P::BM-1)/P::BM, nt=(nn+P::BN-1)/P::BN;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(
  ck_tile::GroupedInt8NtWaspCoreKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}

}
extern "C" int ck_tile_int8_cached_launch_private(unsigned route,unsigned legacy,const void* args,int g,int m,int n,hipStream_t stream){
 switch(route){
 case 0:return ck_tile_int8_cached_legacy_dispatch_private(legacy,args,g,m,n,stream);
 case 1:return run<ck_tile::int8_nt_instances::RawM256N256>(args,g,m,n,stream);
 case 2:return run<ck_tile::int8_nt_instances::RawM256N256Record1024>(args,g,m,n,stream);
 case 3:return run<ck_tile::int8_nt_instances::RawTransposeM256N256>(args,g,m,n,stream);
 case 4:return run<ck_tile::int8_nt_instances::RawTransposeM256N256Record1024>(args,g,m,n,stream);
 case 5:return run<ck_tile::int8_nt_instances::PackedM256N256>(args,g,m,n,stream);
 case 6:return run<ck_tile::int8_nt_instances::PackedM128N256>(args,g,m,n,stream);
 case 7:return run<ck_tile::int8_nt_instances::PackedM128N128>(args,g,m,n,stream);
 case 8:return run<ck_tile::int8_nt_instances::PackedM64N256>(args,g,m,n,stream);
 case 9:return run<ck_tile::int8_nt_instances::PackedM64N64>(args,g,m,n,stream);
 case 10:return run<ck_tile::int8_nt_instances::PackedM32N256>(args,g,m,n,stream);
 case 11:return run<ck_tile::int8_nt_instances::PackedM32N64>(args,g,m,n,stream);
 case 12:return run<ck_tile::int8_nt_instances::PackedM256N384>(args,g,m,n,stream);
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
