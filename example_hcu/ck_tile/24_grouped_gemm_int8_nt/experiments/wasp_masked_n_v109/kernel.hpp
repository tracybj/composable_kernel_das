// SPDX-License-Identifier: MIT
#pragma once
#include "epilogue.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtWaspMaskedNKernel {
 CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const {
  const auto o=args[blockIdx.z].group_karg;if(o.M<=0 || o.N<=0 || o.K<=0)return;
  const UniversalGemmKernelArgs<> a{P::Transpose?o.bs_ptr:o.as_ptr,P::Transpose?o.as_ptr:o.bs_ptr,o.ds_ptr,o.e_ptr,
   P::Transpose?o.N:o.M,P::Transpose?o.M:o.N,o.K,P::Transpose?o.stride_Bs:o.stride_As,P::Transpose?o.stride_As:o.stride_Bs,o.stride_Ds,o.stride_E,o.k_batch};
  const index_t nt=(a.N+P::BN-1)/P::BN,cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
  const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*256,nb=(inner/P::MGroup)*P::BN;
  if(mb>=a.M || nb>=a.N)return;
  __shared__ __align__(16) int8_t smem[65536];
  const bool native=a.M%256==0 && a.N%32==0 && a.K%128==0 && a.stride_Bs[0]>0 && a.stride_Bs[0]<16384 && 256LL*a.stride_As[0]<=0x7fffffffLL && static_cast<long long>(a.N-nb)*a.stride_Bs[0]<=0x7fffffffLL;
  if(!native){
   // Complete scalar fallback for diagnostics;production selection can keep
   // the existing provider outside this experiment's native domain.
   const index_t vm=min(index_t(256),a.M-mb),vn=min(index_t(P::BN),a.N-nb);
   for(index_t t=get_thread_id();t<vm*vn;t+=P::Threads){
    const index_t row=mb+t/vn,col=nb+t%vn;int32_t sum=0;
    const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(row)*a.stride_As[0];const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0]);
    for(index_t k=0;k<a.K;++k){
     const long long bo=P::PackedB?(((static_cast<long long>(k/64)*(a.N/16)+col/16)*4+(k%64)/16)*16+col%16)*16+k%16:static_cast<long long>(col)*a.stride_Bs[0]+k;
     sum+=int32_t(ap[k])*int32_t(bp[bo]);
    }
    const long long co=P::Transpose?static_cast<long long>(col)*a.stride_E+row:static_cast<long long>(row)*a.stride_E+col;static_cast<int32_t*>(a.e_ptr)[co]=sum;
   }return;
  }
  typename P::Pipeline::AccTile acc;const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*(P::PackedB?64:a.stride_Bs[0]);
  typename P::Pipeline{}.run_raw_into(ap,a.stride_As[0],256,bp,a.stride_Bs[0],a.N,nb,a.K/128,acc,smem);
  GroupedInt8NtWaspMaskedNEpilogue<P>{}(a,mb,nb,acc);
 }
};
}
