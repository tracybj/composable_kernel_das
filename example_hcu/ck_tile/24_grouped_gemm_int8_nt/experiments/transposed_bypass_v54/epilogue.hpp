// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
// Internal D[N,M]=B[N,K]*A[M,K]^T writes the external row-major C[M,N].
// Adjacent lane-M coordinates become adjacent external output columns.
template<class P> struct GroupedInt8NtTransposedEpilogue
{
    using Acc=typename P::Mmac::CVecType;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,index_t mb,index_t nb,
                                   const Acc (&c)[P::MR][P::NR]) const
    {
        const index_t lane=get_thread_id()%64,wave=get_thread_id()/64;
        if(wave>=P::WM*P::WN)return;
        const index_t wm=wave/P::WN,wn=wave%P::WN;
        static_for<0,P::MR,1>{}([&](auto i){
            constexpr index_t mi=decltype(i)::value;
            const index_t row=mb+wm*(P::BM/P::WM)+mi*16+lane%16;
            static_for<0,P::NR,1>{}([&](auto j){
                constexpr index_t ni=decltype(j)::value;
                const index_t base=nb+wn*(P::BN/P::WN)+ni*16+lane/16;
                static_for<0,4,1>{}([&](auto v){
                    constexpr index_t vi=decltype(v)::value;const index_t col=base+vi*4;
                    if(P::Aligned || (row<a.M && col<a.N))
                        static_cast<int32_t*>(a.e_ptr)[static_cast<long long>(col)*a.stride_E+row]=c[mi][ni][vi];
                });
            });
        });
    }
};
}
