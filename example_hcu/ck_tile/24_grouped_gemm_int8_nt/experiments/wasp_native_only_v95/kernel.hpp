// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"

namespace ck_tile {
template<class P> struct GroupedInt8NtWaspNativeOnlyKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto o=args[blockIdx.z].group_karg;if(o.M<=0 || o.N<=0 || o.K<=0)return;
        const UniversalGemmKernelArgs<> a{
            P::Transpose?o.bs_ptr:o.as_ptr,P::Transpose?o.as_ptr:o.bs_ptr,o.ds_ptr,o.e_ptr,
            P::Transpose?o.N:o.M,P::Transpose?o.M:o.N,o.K,
            P::Transpose?o.stride_Bs:o.stride_As,P::Transpose?o.stride_As:o.stride_Bs,
            o.stride_Ds,o.stride_E,o.k_batch};
        const index_t nt=(a.N+255)/256,cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*256,nb=(inner/P::MGroup)*256;
        if(mb>=a.M || nb>=a.N)return;
        __shared__ __align__(16) int8_t smem[65536];
        const bool native=a.M%256==0 && a.N%256==0 && a.K%128==0 &&
            a.stride_Bs[0]>0 && a.stride_Bs[0]<16384 &&
            256LL*a.stride_As[0]<=0x7fffffffLL && static_cast<long long>(a.N-nb)*a.stride_Bs[0]<=0x7fffffffLL;
        if(!native)
        {
            // Unsupported descriptors must never select this diagnostic-only kernel.
            return;
        }
        typename P::Pipeline::AccTile acc;
        const auto* ap=static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(mb)*a.stride_As[0];
        const auto* bp=static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(nb)*a.stride_Bs[0];
        typename P::Pipeline{}.run_raw_into(ap,a.stride_As[0],256,bp,a.stride_Bs[0],a.N,nb,a.K/128,acc,smem);
        GroupedInt8NtWaspLinearEpilogue<P>{}(a,mb,nb,acc);
    }
};
}
