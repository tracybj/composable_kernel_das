// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_pipeline.hpp"

namespace ck_tile {
// Cooperative load-next/compute-current with private input registers. Each
// K64 stripe is consumed as two interleaved K32 sets; both operands use exactly
// the same K permutation. No wave exchange is needed for the 16B lane vector.
template <class Base, bool FullNK_ = false, bool Address32_ = false>
struct GroupedInt8NtPackedSmallProblem : Base
{
    static constexpr bool FullNK = FullNK_, Address32 = Address32_;
    static_assert(!Address32 || FullNK, "32-bit offsets require the complete N/K domain");
};

template <class Base, bool FullNK, bool Address32>
struct GroupedInt8NtTilePolicy<GroupedInt8NtPackedSmallProblem<Base, FullNK, Address32>>
    : GroupedInt8NtTilePolicy<Base>
{
    using Parent                          = GroupedInt8NtTilePolicy<Base>;
    static constexpr index_t OperandBytes = 0, OperandAllocation = 0,
                             SmemBytes = Parent::CStripeBytes;
};

template <class Base, bool FullNK, bool Address32, class Policy>
struct GroupedInt8NtTilePipeline<GroupedInt8NtPackedSmallProblem<Base, FullNK, Address32>, Policy>
{
    using P      = GroupedInt8NtPackedSmallProblem<Base, FullNK, Address32>;
    using Mmac   = typename P::Mmac;
    using Copy   = typename P::CopyVector;
    using Layout = Int8NtPackedWeightLayout;
    using AVec   = typename Mmac::AVecType;
    using BVec   = typename Mmac::BVecType;
    using Acc    = typename Mmac::CVecType;
    struct GlobalTile
    {
        Copy a[P::MR][P::BK / Layout::KTile], b[P::NR][P::BK / Layout::KTile];
    };

    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   index_t kb) const
    {
        GlobalTile v{};
        const index_t tid = get_thread_id(), lane = tid % P::WaveSize, wave = tid / P::WaveSize;
        const index_t wm = wave / P::WN, wn = wave % P::WN, q = lane / P::MmacRows;
        static_for<0, P::MR, 1>{}([&](auto i) {
            static_for<0, P::BK / Layout::KTile, 1>{}([&](auto j) {
                constexpr index_t mi = decltype(i)::value, ki = decltype(j)::value;
                const index_t row =
                                  mb + wm * (P::BM / P::WM) + mi * P::MmacRows + lane % P::MmacRows,
                              k = kb + ki * Layout::KTile + q * Layout::VectorBytes;
                if(row < arg.M && (P::FullNK || k + P::InputVectorBytes <= arg.K))
                {
                    // The public plan checks each A matrix span against INT32_MAX.
                    // Only the byte displacement is narrowed; the base stays 64-bit.
                    if constexpr(P::Address32)
                    {
                        const uint32_t offset =
                            static_cast<uint32_t>(row) * static_cast<uint32_t>(arg.stride_As[0]) +
                            static_cast<uint32_t>(k);
                        v.a[mi][ki] = *reinterpret_cast<const Copy*>(
                            static_cast<const int8_t*>(arg.as_ptr[0]) + offset);
                    }
                    else
                    {
                        v.a[mi][ki] = *reinterpret_cast<const Copy*>(
                            static_cast<const int8_t*>(arg.as_ptr[0]) +
                            static_cast<long long>(row) * arg.stride_As[0] + k);
                    }
                }
            });
        });
        static_for<0, P::NR, 1>{}([&](auto i) {
            static_for<0, P::BK / Layout::KTile, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(i)::value, ki = decltype(j)::value;
                const index_t row =
                                  nb + wn * (P::BN / P::WN) + ni * P::MmacCols + lane % P::MmacCols,
                              k = kb + ki * Layout::KTile + q * Layout::VectorBytes;
                if(P::FullNK || (row < arg.N && k + P::InputVectorBytes <= arg.K))
                {
                    // Existing prepared layout:[K/64,N/16,4,16,16]. A wave reads
                    // one contiguous 1024-byte N16/K64 block; raw activation stays MK.
                    if constexpr(P::Address32)
                    {
                        // Prepared B has exactly N*K bytes per group. Host rejects
                        // spans outside INT32_MAX before enabling this instance.
                        const uint32_t offset =
                            Layout::vector_offset(static_cast<uint32_t>(kb / Layout::KTile + ki),
                                                  static_cast<uint32_t>(arg.N / Layout::NTile),
                                                  static_cast<uint32_t>(row / Layout::NTile),
                                                  static_cast<uint32_t>(lane));
                        v.b[ni][ki] = *reinterpret_cast<const Copy*>(
                            static_cast<const int8_t*>(arg.bs_ptr[0]) + offset);
                    }
                    else
                    {
                        const long long offset =
                            Layout::vector_offset(static_cast<long long>(kb / Layout::KTile + ki),
                                                  static_cast<long long>(arg.N / Layout::NTile),
                                                  static_cast<long long>(row / Layout::NTile),
                                                  static_cast<long long>(lane));
                        v.b[ni][ki] = *reinterpret_cast<const Copy*>(
                            static_cast<const int8_t*>(arg.bs_ptr[0]) + offset);
                    }
                }
            });
        });
        return v;
    }

    CK_TILE_DEVICE void Compute(const GlobalTile& v, Acc (&c)[P::MR][P::NR]) const
    {
        static_for<0, P::BK / Layout::KTile, 1>{}([&](auto k) {
            static_for<0, 2, 1>{}([&](auto part) {
                constexpr index_t ki = decltype(k)::value, p = decltype(part)::value;
                AVec av[P::MR];
                BVec bv[P::NR];
                static_for<0, P::MR, 1>{}([&](auto i) {
                    constexpr index_t mi = decltype(i)::value;
                    const auto x         = v.a[mi][ki];
                    av[mi] = bit_cast<AVec>(ext_vector_t<int32_t, 2>{x[p * 2], x[p * 2 + 1]});
                });
                static_for<0, P::NR, 1>{}([&](auto i) {
                    constexpr index_t ni = decltype(i)::value;
                    const auto x         = v.b[ni][ki];
                    bv[ni] = bit_cast<BVec>(ext_vector_t<int32_t, 2>{x[p * 2], x[p * 2 + 1]});
                });
                static_for<0, P::MR, 1>{}([&](auto i) {
                    static_for<0, P::NR, 1>{}([&](auto j) {
                        Mmac{}(c[decltype(i)::value][decltype(j)::value],
                               av[decltype(i)::value],
                               bv[decltype(j)::value]);
                    });
                });
            });
        });
    }

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   Acc (&c)[P::MR][P::NR]) const
    {
        (void)smem;
        auto current = Load(arg, mb, nb, 0);
        for(index_t kb = 0; kb + P::BK < arg.K; kb += P::BK)
        {
            auto next = Load(arg, mb, nb, kb + P::BK);
            __builtin_amdgcn_sched_barrier(0);
            Compute(current, c);
            __builtin_amdgcn_sched_barrier(0);
            current = next;
        }
        Compute(current, c);
        // The existing epilogue still exchanges C through LDS. Only input
        // operands and their main-loop CTA barriers are removed by this policy.
    }
};
} // namespace ck_tile
