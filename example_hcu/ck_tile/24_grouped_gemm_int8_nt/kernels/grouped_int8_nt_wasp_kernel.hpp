// SPDX-License-Identifier: MIT
#pragma once

#include "grouped_int8_nt_pipeline.hpp"
#include "grouped_int8_nt_wasp_problem.hpp"
#include "grouped_int8_nt_epilogue.hpp"
#include "grouped_int8_nt_scaled_epilogue.hpp"

namespace ck_tile {

// -----------------------------------------------------------------------------
// 1. Cooperative LDS Fallback for 256x256 Unpacked WASP outside native stride domain
// -----------------------------------------------------------------------------
struct GroupedInt8NtWaspFallbackProblem
    : GroupedInt8NtTileProblem<256, 256, 64, 4, 2, false, false>
{
    static constexpr index_t Threads = 768, ComputeThreads = 512;
};

template <class P>
struct GroupedInt8NtWaspFallback
{
    CK_TILE_DEVICE void
    operator()(const UniversalGemmKernelArgs<>& a, index_t mb, index_t nb, int8_t* smem) const
    {
        using F    = GroupedInt8NtWaspFallbackProblem;
        using Base = GroupedInt8NtTilePipeline<F>;
        static_assert(GroupedInt8NtTilePolicy<F>::SmemBytes <= 65536);
        typename F::Mmac::CVecType c[F::MR][F::NR]{};
        Base pipeline;
        auto current = pipeline.Load(a, mb, nb, 0);
        pipeline.Stage(current, smem);
        block_sync_lds();
        for(index_t kb = 0; kb < a.K; kb += F::BK)
        {
            typename Base::GlobalTile next{};
            if(kb + F::BK < a.K)
                next = pipeline.Load(a, mb, nb, kb + F::BK);
            if(get_thread_id() < F::ComputeThreads)
                pipeline.Compute(smem, c);
            block_sync_lds();
            if(kb + F::BK < a.K)
            {
                pipeline.Stage(next, smem);
                block_sync_lds();
            }
        }
        const index_t tid = get_thread_id();
        if(tid >= F::ComputeThreads)
            return;
        const index_t lane = tid % 64, wave = tid / 64, wm = wave / F::WN, wn = wave % F::WN;
        static_for<0, F::MR, 1>{}([&](auto mm) {
            constexpr index_t mi = decltype(mm)::value;
            const index_t row    = mb + wm * (F::BM / F::WM) + mi * 16 + lane % 16;
            static_for<0, F::NR, 1>{}([&](auto nn) {
                constexpr index_t ni = decltype(nn)::value;
                static_for<0, 4, 1>{}([&](auto vv) {
                    constexpr index_t vi = decltype(vv)::value;
                    const index_t col    = nb + wn * (F::BN / F::WN) + ni * 16 + lane / 16 + vi * 4;
                    if(row < a.M && col < a.N)
                    {
                        const long long offset =
                            P::Transpose ? static_cast<long long>(col) * a.stride_E + row
                                         : static_cast<long long>(row) * a.stride_E + col;
                        static_cast<int32_t*>(a.e_ptr)[offset] = c[mi][ni][vi];
                    }
                });
            });
        });
    }
};

// -----------------------------------------------------------------------------
// 2. Unified WASP Kernel (supports both Raw INT32 and Fused Scaled BF16/FP16)
// -----------------------------------------------------------------------------
template <class P, bool BFloat = false, class Epilogue = GroupedInt8NtWaspCoreEpilogue<P>>
struct GroupedInt8NtWaspKernel
{
    template <bool AllowTransposedMTail = false>
    CK_TILE_DEVICE static bool is_native(const UniversalGemmKernelArgs<>& a, index_t nb)
    {
        constexpr index_t m_align =
            (AllowTransposedMTail && P::Transpose && P::Pipeline::MasksInactiveN) ? 16 : P::BM;
        constexpr index_t n_align = (P::BN == 384 || P::Pipeline::MasksInactiveN) ? 32 : P::BN;
        return a.M % m_align == 0 && a.N % n_align == 0 && a.K % 128 == 0 &&
               a.stride_Bs[0] > 0 && a.stride_Bs[0] < 16384 &&
               static_cast<long long>(min(index_t(P::BM), a.M)) * a.stride_As[0] <= 0x7fffffffLL &&
               static_cast<long long>(a.N - nb) * a.stride_Bs[0] <= 0x7fffffffLL;
    }

    template <class StoreFn>
    CK_TILE_DEVICE static void
    scalar_fallback(const UniversalGemmKernelArgs<>& a, index_t mb, index_t nb, StoreFn&& store_fn)
    {
        const index_t vm = min(index_t(P::BM), a.M - mb), vn = min(index_t(P::BN), a.N - nb);
        for(index_t t = get_thread_id(); t < vm * vn; t += P::Threads)
        {
            const index_t row = mb + t / vn, col = nb + t % vn;
            int32_t sum    = 0;
            const auto* ap = static_cast<const int8_t*>(a.as_ptr[0]) +
                             static_cast<long long>(row) * a.stride_As[0];
            const auto* bp = static_cast<const int8_t*>(a.bs_ptr[0]);
            for(index_t k = 0; k < a.K; ++k)
            {
                const long long bo =
                    P::PackedB
                        ? (((static_cast<long long>(k / 64) * (a.N / 16) + col / 16) * 4 +
                            (k % 64) / 16) *
                               16 +
                           col % 16) *
                                  16 +
                              k % 16
                        : static_cast<long long>(col) * a.stride_Bs[0] + k;
                sum += int32_t(ap[k]) * int32_t(bp[bo]);
            }
            store_fn(row, col, sum);
        }
    }

    // Raw INT32 entry point
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto o = args[blockIdx.z].group_karg;
        if(o.M <= 0 || o.N <= 0 || o.K <= 0)
            return;
        const UniversalGemmKernelArgs<> a{P::Transpose ? o.bs_ptr : o.as_ptr,
                                          P::Transpose ? o.as_ptr : o.bs_ptr,
                                          o.ds_ptr,
                                          o.e_ptr,
                                          P::Transpose ? o.N : o.M,
                                          P::Transpose ? o.M : o.N,
                                          o.K,
                                          P::Transpose ? o.stride_Bs : o.stride_As,
                                          P::Transpose ? o.stride_As : o.stride_Bs,
                                          o.stride_Ds,
                                          o.stride_E,
                                          o.k_batch};
        const index_t mt = (a.M + P::BM - 1) / P::BM, nt = (a.N + P::BN - 1) / P::BN;
        if(blockIdx.x >= mt * nt)
            return;
        const index_t cluster = blockIdx.x / (P::MGroup * nt),
                      inner   = blockIdx.x % (P::MGroup * nt),
                      mg      = min(index_t(P::MGroup), mt - cluster * P::MGroup);
        const index_t mb      = (cluster * P::MGroup + inner % mg) * P::BM,
                      nb      = (inner / mg) * P::BN;
        if(mb >= a.M || nb >= a.N)
            return;
        __shared__ __align__(16) int8_t smem[P::Pipeline::kWaspLdsBytes];
        if(!is_native<true>(a, nb))
        {
            if constexpr(P::BM == 256 && P::BN == 256 && !P::PackedB)
            {
                GroupedInt8NtWaspFallback<P>{}(a, mb, nb, smem);
            }
            else
            {
                scalar_fallback(a, mb, nb, [&](index_t row, index_t col, int32_t sum) {
                    const long long co = P::Transpose
                                             ? static_cast<long long>(col) * a.stride_E + row
                                             : static_cast<long long>(row) * a.stride_E + col;
                    static_cast<int32_t*>(a.e_ptr)[co] = sum;
                });
            }
            return;
        }
        typename P::Pipeline::AccTile acc;
        const auto* ap =
            static_cast<const int8_t*>(a.as_ptr[0]) + static_cast<long long>(mb) * a.stride_As[0];
        const auto* bp = static_cast<const int8_t*>(a.bs_ptr[0]) +
                         static_cast<long long>(nb) * (P::PackedB ? 64 : a.stride_Bs[0]);
        typename P::Pipeline{}.run_raw_into(ap,
                                            a.stride_As[0],
                                            min(index_t(P::BM), a.M - mb),
                                            bp,
                                            a.stride_Bs[0],
                                            a.N,
                                            nb,
                                            a.K / 128,
                                            acc,
                                            smem);
        if constexpr(P::Pipeline::MasksInactiveN)
        {
            const index_t wave = get_warp_id();
            if(wave < P::ProducerWaves || nb + (wave - P::ProducerWaves) * 32 >= a.N)
                return;
        }
        Epilogue{}(a, mb, nb, acc);
    }

    // Fused Scaled (BF16 / FP16) entry point
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,
                                   const Int8NtScaleArgs* scales) const
    {
        const auto o = args[blockIdx.z].group_karg;
        if(o.M <= 0 || o.N <= 0 || o.K <= 0)
            return;
        const UniversalGemmKernelArgs<> a{P::Transpose ? o.bs_ptr : o.as_ptr,
                                          P::Transpose ? o.as_ptr : o.bs_ptr,
                                          o.ds_ptr,
                                          o.e_ptr,
                                          P::Transpose ? o.N : o.M,
                                          P::Transpose ? o.M : o.N,
                                          o.K,
                                          P::Transpose ? o.stride_Bs : o.stride_As,
                                          P::Transpose ? o.stride_As : o.stride_Bs,
                                          o.stride_Ds,
                                          o.stride_E,
                                          o.k_batch};
        const index_t mt = (a.M + P::BM - 1) / P::BM, nt = (a.N + P::BN - 1) / P::BN;
        if(blockIdx.x >= mt * nt)
            return;
        const index_t cluster = blockIdx.x / (P::MGroup * nt),
                      inner   = blockIdx.x % (P::MGroup * nt),
                      mg      = min(index_t(P::MGroup), mt - cluster * P::MGroup);
        const index_t mb      = (cluster * P::MGroup + inner % mg) * P::BM,
                      nb      = (inner / mg) * P::BN;
        if(mb >= a.M || nb >= a.N)
            return;
        __shared__ __align__(16) int8_t smem[P::Pipeline::kWaspLdsBytes];
        if(!is_native<!P::PackedB>(a, nb))
        {
            scalar_fallback(a, mb, nb, [&](index_t row, index_t col, int32_t sum) {
                Int8NtDequantStore<BFloat>{scales[blockIdx.z]}.one(
                    o, P::Transpose ? col : row, P::Transpose ? row : col, sum);
            });
            return;
        }
        typename P::Pipeline::AccTile acc;
        const auto* ap =
            static_cast<const int8_t*>(a.as_ptr[0]) + static_cast<long long>(mb) * a.stride_As[0];
        const auto* bp = static_cast<const int8_t*>(a.bs_ptr[0]) +
                         static_cast<long long>(nb) * (P::PackedB ? 64 : a.stride_Bs[0]);
        typename P::Pipeline{}.run_raw_into(ap,
                                            a.stride_As[0],
                                            min(index_t(P::BM), a.M - mb),
                                            bp,
                                            a.stride_Bs[0],
                                            a.N,
                                            nb,
                                            a.K / 128,
                                            acc,
                                            smem);
        if constexpr(P::Pipeline::MasksInactiveN)
        {
            const index_t wave = get_warp_id();
            if(wave < P::ProducerWaves || nb + (wave - P::ProducerWaves) * 32 >= a.N)
                return;
        }
        Epilogue{}(o, mb, nb, acc, scales[blockIdx.z]);
    }
};

template <class P>
using GroupedInt8NtWaspCoreKernel = GroupedInt8NtWaspKernel<P, false, GroupedInt8NtWaspCoreEpilogue<P>>;

template <class P, bool BFloat, class Epilogue = GroupedInt8NtScaledWaspEpilogue<P, BFloat>>
using GroupedInt8NtScaledWaspKernel = GroupedInt8NtWaspKernel<P, BFloat, Epilogue>;

} // namespace ck_tile
