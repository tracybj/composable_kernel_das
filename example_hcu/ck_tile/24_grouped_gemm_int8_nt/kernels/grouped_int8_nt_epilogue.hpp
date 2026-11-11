// SPDX-License-Identifier: MIT
#pragma once

#include "grouped_int8_nt_policy.hpp"
#include "grouped_int8_nt_wasp_problem.hpp"

namespace ck_tile {

// -----------------------------------------------------------------------------
// 1. Cooperative Epilogue: stages accumulators through LDS before writing to GMEM
// -----------------------------------------------------------------------------
template <class P, class Policy = GroupedInt8NtTilePolicy<P>>
struct GroupedInt8NtTileEpilogue
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
                const index_t row = wm * 16 + lane % 16;
                static_for<0, P::NR, 1>{}([&](auto j) {
                    static_for<0, 4, 1>{}([&](auto v) {
                        constexpr index_t ni = decltype(j)::value, vi = decltype(v)::value;
                        const index_t col = wn * (P::BN / P::WN) + ni * 16 + lane / 16 + vi * 4;
                        tmp[row * Policy::CPitch + (col ^ ((row % 8) * 4))] = c[mi][ni][vi];
                    });
                });
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

// -----------------------------------------------------------------------------
// 2. Direct Epilogue: register-only transpose via bpermute without LDS staging
// -----------------------------------------------------------------------------
template <class P, bool Vector = true>
struct GroupedInt8NtDirectEpilogue
{
    using Acc = typename P::Mmac::CVecType;
    CK_TILE_DEVICE static int32_t exchange(int32_t value, index_t source_lane)
    {
        return __builtin_amdgcn_ds_bpermute(source_lane * 4, value);
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   int8_t*,
                                   const Acc (&c)[P::MR][P::NR]) const
    {
        (*this)(a, mb, nb, c);
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   const Acc (&c)[P::MR][P::NR]) const
    {
        const index_t tid = get_thread_id(), lane = tid % 64, wave = tid / 64;
        const index_t wm = wave / P::WN, wn = wave % P::WN, h = lane / 16;
        if(wm >= P::WM)
            return;
        static_for<0, P::MR, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            const index_t row    = mb + wm * (P::BM / P::WM) + mi * 16 + lane % 16;
            static_for<0, P::NR, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(j)::value;
                const index_t base   = nb + wn * (P::BN / P::WN) + ni * 16;
                if constexpr(Vector)
                {
                    int32_t r0 = c[mi][ni][0], r1 = c[mi][ni][1], r2 = c[mi][ni][2],
                            r3 = c[mi][ni][3];
                    // Transpose the 4x4 (lane-K-group, accumulator-component) matrix.
                    const bool b0 = (h & 1) != 0, b1 = (h & 2) != 0;
                    int32_t t0        = exchange(b0 ? r0 : r1, lane ^ 16),
                            t1        = exchange(b0 ? r2 : r3, lane ^ 16);
                    r0                = b0 ? t0 : r0;
                    r1                = b0 ? r1 : t0;
                    r2                = b0 ? t1 : r2;
                    r3                = b0 ? r3 : t1;
                    t0                = exchange(b1 ? r0 : r2, lane ^ 32);
                    t1                = exchange(b1 ? r1 : r3, lane ^ 32);
                    r0                = b1 ? t0 : r0;
                    r2                = b1 ? r2 : t0;
                    r1                = b1 ? t1 : r1;
                    r3                = b1 ? r3 : t1;
                    const index_t col = base + h * 4;
                    if(P::Aligned || (row < a.M && col + 4 <= a.N))
                    {
                        auto* ptr = static_cast<int32_t*>(a.e_ptr) +
                                    static_cast<long long>(row) * a.stride_E + col;
                        if((a.stride_E & 3) == 0 &&
                           (reinterpret_cast<uintptr_t>(a.e_ptr) & 15) == 0)
                            *reinterpret_cast<typename P::CopyVector*>(ptr) =
                                typename P::CopyVector{r0, r1, r2, r3};
                        else
                        {
                            ptr[0] = r0;
                            ptr[1] = r1;
                            ptr[2] = r2;
                            ptr[3] = r3;
                        }
                    }
                }
                else
                {
                    static_for<0, 4, 1>{}([&](auto v) {
                        constexpr index_t vi = decltype(v)::value;
                        const index_t col    = base + h + vi * 4;
                        if(P::Aligned || (row < a.M && col < a.N))
                            static_cast<int32_t*>(
                                a.e_ptr)[static_cast<long long>(row) * a.stride_E + col] =
                                c[mi][ni][vi];
                    });
                }
            });
        });
    }
};

// -----------------------------------------------------------------------------
// 3. WASP Epilogue: Producer-Consumer decoupled INT32 output with wide-N masking
// -----------------------------------------------------------------------------
template <class P>
struct GroupedInt8NtWaspCoreEpilogue
{
    using Tile = typename P::Pipeline::AccTile;
    CK_TILE_DEVICE void
    operator()(const UniversalGemmKernelArgs<>& a, index_t mb, index_t nb, const Tile& tile) const
    {
        const index_t tid = get_thread_id(), lane = tid % 64, wave = tid / 64;
        if(wave < P::ProducerWaves)
            return;
        const index_t consumer = wave - P::ProducerWaves, h = lane / 16;
        if constexpr(P::BN == 384 || P::Pipeline::MasksInactiveN)
            if(nb + consumer * 32 >= a.N)
                return; // Whole 32-column consumer stripe outside N.
        static_for<0, P::BM / 16, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            if constexpr(P::Transpose && P::Pipeline::MasksInactiveN)
                if(mb + mi * 16 >= a.M)
                    return;
            const index_t row    = mb + mi * 16 + lane % 16;
            static_for<0, 2, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(j)::value;
                const auto v         = tile.v[mi * 2 + ni];
                const index_t base   = nb + consumer * 32 + ni * 16;
                if constexpr(P::Transpose)
                {
                    static_for<0, 4, 1>{}([&](auto k) {
                        constexpr index_t vi = decltype(k)::value;
                        const index_t col    = base + h + vi * 4;
                        static_cast<int32_t*>(
                            a.e_ptr)[static_cast<long long>(col) * a.stride_E + row] = v[vi];
                    });
                }
                else
                {
                    int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                    const bool b0 = h & 1, b1 = h & 2;
                    int32_t t0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1);
                    int32_t t1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
                    r0         = b0 ? t0 : r0;
                    r1         = b0 ? r1 : t0;
                    r2         = b0 ? t1 : r2;
                    r3         = b0 ? r3 : t1;
                    t0         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r2);
                    t1         = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r1 : r3);
                    r0         = b1 ? t0 : r0;
                    r2         = b1 ? r2 : t0;
                    r1         = b1 ? t1 : r1;
                    r3         = b1 ? r3 : t1;
                    auto* ptr  = static_cast<int32_t*>(a.e_ptr) +
                                static_cast<long long>(row) * a.stride_E + base + h * 4;
                    if((a.stride_E & 3) == 0 && (reinterpret_cast<uintptr_t>(a.e_ptr) & 15) == 0)
                        *reinterpret_cast<typename P::CopyVector*>(ptr) =
                            typename P::CopyVector{r0, r1, r2, r3};
                    else
                    {
                        ptr[0] = r0;
                        ptr[1] = r1;
                        ptr[2] = r2;
                        ptr[3] = r3;
                    }
                }
            });
        });
    }
};

// Typedef alias for standard WASP pipelines
template <class P>
using GroupedInt8NtWaspEpilogue = GroupedInt8NtWaspCoreEpilogue<P>;

} // namespace ck_tile
