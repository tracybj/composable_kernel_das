// SPDX-License-Identifier: MIT
#pragma once
#include "scaled_epilogue.hpp"
#include "core_problem.hpp"
#include "../grouped_int8_nt_pipeline.hpp"
namespace ck_tile {
template<class P,bool BFloat>
struct GroupedInt8NtScaledTileKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,const Int8NtScaleArgs* scales) const
    {
        const index_t tid = get_thread_id();
        __builtin_assume(tid < P::Threads);
        using ArgStorage = std::
            conditional_t<P::CopyArg, UniversalGemmKernelArgs<>, const UniversalGemmKernelArgs<>&>;
        const ArgStorage a = args[blockIdx.z].group_karg;
        index_t mb, nb;
        if constexpr(P::MGroup > 0)
        {
            if(a.M <= 0 || a.N <= 0 || a.K <= 0)
                return;
            const index_t nt      = (a.N + P::BN - 1) / P::BN;
            const index_t linear  = blockIdx.x;
            const index_t cluster = linear / (P::MGroup * nt);
            const index_t inner   = linear % (P::MGroup * nt);
            mb                    = (cluster * P::MGroup + inner % P::MGroup) * P::BM;
            nb                    = (inner / P::MGroup) * P::BN;
            if(mb >= a.M || nb >= a.N)
                return;
        }
        else
        {
            mb = (P::MFirst ? blockIdx.x : blockIdx.y) * P::BM;
            nb = (P::MFirst ? blockIdx.y : blockIdx.x) * P::BN;
        }
        if constexpr(!P::Aligned)
            if(a.M <= 0 || a.N <= 0 || a.K <= 0 || mb >= a.M || nb >= a.N)
                return;
        using Policy = GroupedInt8NtTilePolicy<P>;
        __shared__ __align__(16) int8_t smem[Policy::SmemBytes];
        typename P::Mmac::CVecType c[P::MR][P::NR]{};
        GroupedInt8NtTilePipeline<P>{}(a, mb, nb, smem, c);
        GroupedInt8NtScaledTileEpilogue<P,BFloat>{}(a,mb,nb,smem,c,scales[blockIdx.z]);
    }
};
} // namespace ck_tile

namespace ck_tile {
template<class P,bool BFloat,class Epilogue=GroupedInt8NtScaledWaspEpilogue<P,BFloat>> struct GroupedInt8NtScaledWaspKernel {
 CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,const Int8NtScaleArgs* scales) const {
  const auto o=args[blockIdx.z].group_karg;if(o.M<=0 || o.N<=0 || o.K<=0)return;
  const UniversalGemmKernelArgs<> a{P::Transpose?o.bs_ptr:o.as_ptr,P::Transpose?o.as_ptr:o.bs_ptr,o.ds_ptr,o.e_ptr,
   P::Transpose?o.N:o.M,P::Transpose?o.M:o.N,o.K,P::Transpose?o.stride_Bs:o.stride_As,P::Transpose?o.stride_As:o.stride_Bs,o.stride_Ds,o.stride_E,o.k_batch};
  const index_t nt=(a.N+P::BN-1)/P::BN,cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
  const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM,nb=(inner/P::MGroup)*P::BN;
  if(mb>=a.M || nb>=a.N)return;
  __shared__ __align__(16) int8_t smem[P::Pipeline::kWaspLdsBytes];
  const bool native=a.M%P::BM==0 && a.N%(P::BN==384?32:P::BN)==0 && a.K%128==0 && a.stride_Bs[0]>0 && a.stride_Bs[0]<16384 && static_cast<long long>(P::BM)*a.stride_As[0]<=0x7fffffffLL && static_cast<long long>(a.N-nb)*a.stride_Bs[0]<=0x7fffffffLL;
  if(!native){
   // Complete scalar fallback for diagnostics;production selection can keep
   // the existing provider outside this experiment's native domain.
   const index_t vm=min(index_t(P::BM),a.M-mb),vn=min(index_t(P::BN),a.N-nb);
   for(index_t t=get_thread_id();t<vm*vn;t+=P::Threads){
    const index_t row=mb+t/vn,col=nb+t%vn;int32_t sum=0;
    const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(row)*a.stride_As[0];const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0]);
    for(index_t k=0;k<a.K;++k){
     const long long bo=P::PackedB?(((static_cast<long long>(k/64)*(a.N/16)+col/16)*4+(k%64)/16)*16+col%16)*16+k%16:static_cast<long long>(col)*a.stride_Bs[0]+k;
     sum+=int32_t(ap[k])*int32_t(bp[bo]);
    }
    const long long co=P::Transpose?static_cast<long long>(col)*a.stride_E+row:static_cast<long long>(row)*a.stride_E+col;(void)co;Int8NtDequantStore<BFloat>{scales[blockIdx.z]}.one(o,P::Transpose?col:row,P::Transpose?row:col,sum);
   }return;
  }
  typename P::Pipeline::AccTile acc;const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*(P::PackedB?64:a.stride_Bs[0]);
  typename P::Pipeline{}.run_raw_into(ap,a.stride_As[0],P::BM,bp,a.stride_Bs[0],a.N,nb,a.K/128,acc,smem);
  Epilogue{}(o,mb,nb,acc,scales[blockIdx.z]);
 }
};
}
