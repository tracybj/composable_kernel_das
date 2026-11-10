// SPDX-License-Identifier: MIT
#pragma once
#include "../../cached_weights/kernels/scaled_epilogue.hpp"
#include "../../kernels/grouped_int8_nt_epilogue.hpp"
#include "../../pipeline/packed_cooperative.hpp"

namespace ck_tile {
// Experimental policy: preserve the input pipeline and change only C exchange.
template <class Base>
struct GroupedInt8NtWaveStoreProblem : Base
{
    static_assert(Base::BM == 16 && Base::WM == 1);
};

template <class Base>
struct GroupedInt8NtTilePolicy<GroupedInt8NtWaveStoreProblem<Base>> : GroupedInt8NtTilePolicy<Base>
{
    using Parent = GroupedInt8NtTilePolicy<Base>;
    // C never uses shared storage. The dummy minimum permits a standard shared
    // array declaration; when the input pipeline also uses no LDS it is dead.
    static constexpr index_t SmemBytes =
        Parent::OperandAllocation > 0 ? Parent::OperandAllocation : 16;
};

template <class Base, class Policy>
struct GroupedInt8NtTilePipeline<GroupedInt8NtWaveStoreProblem<Base>, Policy>
    : GroupedInt8NtTilePipeline<Base, Policy>
{
};

template <class P, bool Scaled, bool BFloat>
struct GroupedInt8NtWaveStoreEpilogue
{
    using Acc = typename P::Mmac::CVecType;
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& a,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   const Acc (&c)[P::MR][P::NR],
                                   Int8NtScaleArgs scale = {}) const
    {
        (void)smem;
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        const index_t wm = wave / P::WN, wn = wave % P::WN, h = lane / 16;
        const bool b0 = h & 1, b1 = h & 2;
        const Int8NtDequantStore<BFloat> store{scale};
        static_for<0, P::MR, 1>{}([&](auto i) {
            constexpr index_t mi = decltype(i)::value;
            const index_t row    = mb + wm * (P::BM / P::WM) + mi * 16 + lane % 16;
            static_for<0, P::NR, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(j)::value;
                auto v               = c[mi][ni];
                int32_t r0 = v[0], r1 = v[1], r2 = v[2], r3 = v[3];
                // Every lane participates before any output bounds predicate.
                int32_t t0        = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r0 : r1);
                int32_t t1        = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, b0 ? r2 : r3);
                r0                = b0 ? t0 : r0;
                r1                = b0 ? r1 : t0;
                r2                = b0 ? t1 : r2;
                r3                = b0 ? r3 : t1;
                t0                = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r0 : r2);
                t1                = __builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, b1 ? r1 : r3);
                r0                = b1 ? t0 : r0;
                r2                = b1 ? r2 : t0;
                r1                = b1 ? t1 : r1;
                r3                = b1 ? r3 : t1;
                const index_t col = nb + wn * (P::BN / P::WN) + ni * 16 + h * 4;
                if(row >= a.M || col >= a.N)
                    return;
                if constexpr(Scaled)
                {
                    if(col + 4 <= a.N)
                        store.four(a, row, col, r0, r1, r2, r3);
                    else
                    {
                        const int32_t values[4] = {r0, r1, r2, r3};
                        static_for<0, 4, 1>{}([&](auto k) {
                            if(col + decltype(k)::value < a.N)
                                store.one(
                                    a, row, col + decltype(k)::value, values[decltype(k)::value]);
                        });
                    }
                }
                else
                {
                    auto* dst = static_cast<int32_t*>(a.e_ptr) +
                                static_cast<long long>(row) * a.stride_E + col;
                    if(col + 4 <= a.N && (reinterpret_cast<uintptr_t>(dst) & 15) == 0)
                        *reinterpret_cast<typename P::CopyVector*>(dst) =
                            typename P::CopyVector{r0, r1, r2, r3};
                    else
                    {
                        const int32_t values[4] = {r0, r1, r2, r3};
                        static_for<0, 4, 1>{}([&](auto k) {
                            if(col + decltype(k)::value < a.N)
                                dst[decltype(k)::value] = values[decltype(k)::value];
                        });
                    }
                }
            });
        });
    }
};

template <class Base, class Policy>
struct GroupedInt8NtTileEpilogue<GroupedInt8NtWaveStoreProblem<Base>, Policy>
    : GroupedInt8NtWaveStoreEpilogue<GroupedInt8NtWaveStoreProblem<Base>, false, false>
{
};

template <class Base, bool BFloat>
struct GroupedInt8NtScaledTileEpilogue<GroupedInt8NtWaveStoreProblem<Base>, BFloat>
    : GroupedInt8NtWaveStoreEpilogue<GroupedInt8NtWaveStoreProblem<Base>, true, BFloat>
{
};
} // namespace ck_tile
