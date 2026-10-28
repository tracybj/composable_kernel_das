// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
#include "../aligned_d2l_v39/pipeline.hpp"
#include "epilogue.hpp"
#include "../direct_lds_v30/kernel.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtDirectEpiKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto a=args[blockIdx.z].group_karg;
        if(a.M<=0 || a.N<=0 || a.K<=0)return;
        const index_t nt=(a.N+P::BN-1)/P::BN;
        const index_t cluster=blockIdx.x/(P::MGroup*nt),inner=blockIdx.x%(P::MGroup*nt);
        const index_t mb=(cluster*P::MGroup+inner%P::MGroup)*P::BM;
        const index_t nb=(inner/P::MGroup)*P::BN;
        if(mb>=a.M || nb>=a.N)return;
        using Policy=GroupedInt8NtD2LAlignedPolicy<P>;
        __shared__ __align__(16) int8_t smem[Policy::SmemBytes];
        typename P::Mmac::CVecType c[P::MR][P::NR]{};
        const bool aligned=a.M%P::BM==0 && a.N%P::BN==0 && a.K%P::BK==0;
        const bool native=static_cast<long long>(P::BM-1)*a.stride_As[0]+a.K<=0x7fffffffLL &&
                          static_cast<long long>(P::BN-1)*a.stride_Bs[0]+a.K<=0x7fffffffLL;
        if(!aligned || (P::NativeOnly && !native))
        {
            using Fallback=GroupedInt8NtD2LGeometryProblem<P::WM,P::WN,P::MGroup,false,P::SwizzleShift>;
            GroupedInt8NtD2LGeometryPipeline<Fallback>{}(a,mb,nb,smem,c);
            GroupedInt8NtNoLdsEpilogue<Fallback,P::VectorOutput>{}(a,mb,nb,c);
        }
        else
        {
            GroupedInt8NtD2LAlignedPipeline<P>{}(a,mb,nb,smem,c);
            GroupedInt8NtNoLdsEpilogue<P,P::VectorOutput>{}(a,mb,nb,c);
        }
    }
};
}
