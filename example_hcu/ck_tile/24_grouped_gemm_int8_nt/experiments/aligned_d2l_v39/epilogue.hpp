// SPDX-License-Identifier: MIT
#pragma once
#include "policy.hpp"
namespace ck_tile {
template <class P, class Policy = GroupedInt8NtD2LAlignedPolicy<P>>
struct GroupedInt8NtD2LAlignedEpilogue
{
    using Acc = typename P::Mmac::CVecType;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   const Acc (&c)[P::MR][P::NR]) const
    {
        const index_t tid = get_thread_id(), lane = tid % 64, wave = tid / 64;
        const index_t wm = wave / P::WN, wn = wave % P::WN;
        auto* tmp = reinterpret_cast<int32_t*>(smem);
        block_sync_lds();
        static_for<0, P::MR, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            if constexpr(P::BatchedEpi)
            {
                if(wm<P::WM)
                {
                    const index_t row = wm * 16 + lane % 16;
                    static_for<0, P::NR, 1>{}([&](auto j) {
                        static_for<0, 4, 1>{}([&](auto v) {
                            constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                            const index_t col = wn * (P::BN / P::WN) + ni * 16 + lane / 16 + vi * 4;
                            tmp[row * Policy::CPitch + (col ^ ((row % 8) * 4))] = c[mi][ni][vi];
                        });
                    });
                }
                block_sync_lds();
                for(index_t vec = tid; vec < Policy::CStripeRows * P::BN / 4; vec += P::Threads)
                {
                    const index_t r = vec / (P::BN / 4), col = (vec % (P::BN / 4)) * 4;
                    const index_t outrow = mb + (r / 16) * (P::BM / P::WM) + mi * 16 + r % 16;
                    if(P::Aligned || (outrow < arg.M && nb + col + 4 <= arg.N))
                        *reinterpret_cast<typename P::CopyVector*>(
                            static_cast<int32_t*>(arg.e_ptr) +
                            static_cast<long long>(outrow) * arg.stride_E + nb + col) =
                            *reinterpret_cast<const typename P::CopyVector*>(
                                tmp + r * Policy::CPitch + (col ^ ((r % 8) * 4)));
                }
                block_sync_lds();
            }
            else
            {
                static_for<0, P::WM, 1>{}([&](auto wmi) {
                    constexpr index_t output_wave_m = decltype(wmi)::value;
                    if(wm == output_wave_m)
                    {
                        static_for<0, P::NR, 1>{}([&](auto j) {
                            static_for<0, 4, 1>{}([&](auto v) {
                                constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                                const index_t col =
                                    wn * (P::BN / P::WN) + ni * 16 + lane / 16 + vi * 4;
                                tmp[(lane % 16) * Policy::CPitch + col] = c[mi][ni][vi];
                            });
                        });
                    }
                    block_sync_lds();
                    for(index_t vec = tid; vec < Policy::CStripeRows * P::BN / 4; vec += P::Threads)
                    {
                        const index_t row = vec / (P::BN / 4), col = (vec % (P::BN / 4)) * 4;
                        const index_t outrow = mb + output_wave_m * (P::BM / P::WM) + mi * 16 + row;
                        if(P::Aligned || (outrow < arg.M && nb + col + 4 <= arg.N))
                            *reinterpret_cast<typename P::CopyVector*>(
                                static_cast<int32_t*>(arg.e_ptr) +
                                static_cast<long long>(outrow) * arg.stride_E + nb + col) =
                                *reinterpret_cast<const typename P::CopyVector*>(
                                    tmp + row * Policy::CPitch + col);
                    }
                    block_sync_lds();
                });
            }
        });
    }
};
} // namespace ck_tile
