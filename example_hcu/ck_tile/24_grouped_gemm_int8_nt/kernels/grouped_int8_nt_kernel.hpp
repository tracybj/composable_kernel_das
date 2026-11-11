// SPDX-License-Identifier: MIT
#pragma once

#include "grouped_int8_nt_pipeline.hpp"
#include "grouped_int8_nt_epilogue.hpp"
#include "grouped_int8_nt_scaled_epilogue.hpp"

namespace ck_tile {

template <class P, class Epilogue = GroupedInt8NtTileEpilogue<P>>
struct GroupedInt8NtTileKernel
{
    template <class EpiFn>
    CK_TILE_DEVICE static void run_tile(const GemmTransKernelArg* args, EpiFn&& epi_fn)
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
        epi_fn(a, mb, nb, smem, c);
    }

    // Raw INT32 entry point
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        run_tile(args, [](const auto& a, index_t mb, index_t nb, int8_t* smem, const auto& c) {
            Epilogue{}(a, mb, nb, smem, c);
        });
    }

    // Fused Scaled (BF16 / FP16) entry point
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,
                                   const Int8NtScaleArgs* scales) const
    {
        run_tile(
            args, [scales](const auto& a, index_t mb, index_t nb, int8_t* smem, const auto& c) {
                Epilogue{}(a, mb, nb, smem, c, scales[blockIdx.z]);
            });
    }
};

template <class P, bool BFloat>
using GroupedInt8NtScaledTileKernel =
    GroupedInt8NtTileKernel<P, GroupedInt8NtScaledTileEpilogue<P, BFloat>>;

} // namespace ck_tile
