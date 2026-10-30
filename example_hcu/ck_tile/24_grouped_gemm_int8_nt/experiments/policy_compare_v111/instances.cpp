// SPDX-License-Identifier: MIT
// Reuse the frozen production dispatcher;probe IDs are internal route IDs,
// not requirement case IDs. This adapter does not change production selection.
#include "../../cached_weights/cached_dispatch.cpp"
#include "../wasp_wide_n_v107/kernel.hpp"
namespace {
int run_wide(const void* args,int g,int m,int n,hipStream_t stream){
 using P=ck_tile::GroupedInt8NtWaspWideNProblem<false,12,4,2,true,true,0,false,true,0,12>;
 const int mt=(m+255)/256,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+3)/4)*4*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspWideNKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){
 if(id>=50001 && id<=50009)return ck_tile_int8_cached_launch_private(id-50000,0,args,g,m,n,stream);
 if(id==39002)return run_wide(args,g,m,n,stream);
 return ck_tile_int8_cached_legacy_dispatch_private(id,args,g,m,n,stream);
}
