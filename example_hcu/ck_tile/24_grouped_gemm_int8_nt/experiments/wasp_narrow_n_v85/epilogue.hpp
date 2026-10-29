// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtWaspNarrowEpilogue
{
    using Tile=typename P::Pipeline::AccTile;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,const Tile& tile) const
    {
        const index_t tid=get_thread_id(),lane=tid%64,wave=tid/64;
        if(wave<4)return;
        const index_t consumer=wave-4,h=lane/16;
        static_for<0,Tile::kMIters,1>{}([&](auto i){
            constexpr index_t mi=decltype(i)::value;const index_t row=mb+mi*16+lane%16;
            static_for<0,Tile::kNIters,1>{}([&](auto j){
                constexpr index_t ni=decltype(j)::value;const auto v=tile.v[mi*Tile::kNIters+ni];
                const index_t base=nb+consumer*16+ni*16;
                if constexpr(P::Transpose)
                {
                    static_for<0,4,1>{}([&](auto k){
                        constexpr index_t vi=decltype(k)::value;const index_t col=base+h+vi*4;
                        static_cast<int32_t*>(a.e_ptr)[static_cast<long long>(col)*a.stride_E+row]=v[vi];
                    });
                }
                else
                {
                    int32_t r0=v[0],r1=v[1],r2=v[2],r3=v[3];
                    const bool b0=h&1,b1=h&2;
                    int32_t t0=__builtin_amdgcn_ds_bpermute((lane^16)*4,b0?r0:r1);
                    int32_t t1=__builtin_amdgcn_ds_bpermute((lane^16)*4,b0?r2:r3);
                    r0=b0?t0:r0;r1=b0?r1:t0;r2=b0?t1:r2;r3=b0?r3:t1;
                    t0=__builtin_amdgcn_ds_bpermute((lane^32)*4,b1?r0:r2);
                    t1=__builtin_amdgcn_ds_bpermute((lane^32)*4,b1?r1:r3);
                    r0=b1?t0:r0;r2=b1?r2:t0;r1=b1?t1:r1;r3=b1?r3:t1;
                    auto* ptr=static_cast<int32_t*>(a.e_ptr)+static_cast<long long>(row)*a.stride_E+base+h*4;
                    if((a.stride_E&3)==0 && (reinterpret_cast<uintptr_t>(a.e_ptr)&15)==0)
                        *reinterpret_cast<typename P::CopyVector*>(ptr)=typename P::CopyVector{r0,r1,r2,r3};
                    else {ptr[0]=r0;ptr[1]=r1;ptr[2]=r2;ptr[3]=r3;}
                }
            });
        });
    }
};
}
