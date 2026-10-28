// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
// MMAC fragment -> row-major output, without an LDS exchange or CTA barrier.
template<class P> struct GroupedInt8NtStreamEpilogue
{
    using Acc=typename P::Mmac::CVecType;
    CK_TILE_DEVICE static int32_t exchange(int32_t value,index_t source_lane)
    {
        return __builtin_amdgcn_ds_bpermute(source_lane*4,value);
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,
                                   const Acc (&c)[P::MR][P::NR],index_t logical_wave) const
    {
        const index_t tid=get_thread_id(),lane=tid%64,wave=logical_wave;
        const index_t wm=wave/P::WN,wn=wave%P::WN,h=lane/16;
        if(wm>=P::WM)return;
        static_for<0,P::MR,1>{}([&](auto i){
            constexpr index_t mi=decltype(i)::value;
            const index_t row=mb+wm*(P::BM/P::WM)+mi*16+lane%16;
            static_for<0,P::NR,1>{}([&](auto j){
                constexpr index_t ni=decltype(j)::value;
                const index_t base=nb+wn*(P::BN/P::WN)+ni*16;
                if constexpr(true)
                {
                    int32_t r0=c[mi][ni][0],r1=c[mi][ni][1],r2=c[mi][ni][2],r3=c[mi][ni][3];
                    // Transpose the 4x4 (lane-K-group, accumulator-component) matrix.
                    const bool b0=(h&1)!=0,b1=(h&2)!=0;
                    int32_t t0=exchange(b0?r0:r1,lane^16),t1=exchange(b0?r2:r3,lane^16);
                    r0=b0?t0:r0;r1=b0?r1:t0;r2=b0?t1:r2;r3=b0?r3:t1;
                    t0=exchange(b1?r0:r2,lane^32);t1=exchange(b1?r1:r3,lane^32);
                    r0=b1?t0:r0;r2=b1?r2:t0;r1=b1?t1:r1;r3=b1?r3:t1;
                    const index_t col=base+h*4;
                    if(P::Aligned || (row<a.M && col+4<=a.N))
                    {
                        auto* ptr=static_cast<int32_t*>(a.e_ptr)+static_cast<long long>(row)*a.stride_E+col;
                        if((a.stride_E&3)==0 && (reinterpret_cast<uintptr_t>(a.e_ptr)&15)==0)
                            *reinterpret_cast<typename P::CopyVector*>(ptr)=typename P::CopyVector{r0,r1,r2,r3};
                        else {ptr[0]=r0;ptr[1]=r1;ptr[2]=r2;ptr[3]=r3;}
                    }
                }
                else
                {
                    static_for<0,4,1>{}([&](auto v){
                        constexpr index_t vi=decltype(v)::value;
                        const index_t col=base+h+vi*4;
                        if(P::Aligned || (row<a.M && col<a.N))
                            static_cast<int32_t*>(a.e_ptr)[static_cast<long long>(row)*a.stride_E+col]=c[mi][ni][vi];
                    });
                }
            });
        });
    }
};
}
