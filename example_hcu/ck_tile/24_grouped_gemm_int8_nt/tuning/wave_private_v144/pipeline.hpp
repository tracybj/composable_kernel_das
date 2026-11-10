// SPDX-License-Identifier: MIT
#pragma once
#include "../../kernels/grouped_int8_nt_pipeline.hpp"
namespace ck_tile {
template <class Base>
struct GroupedInt8NtWavePrivateProblem : Base
{
    static_assert(Base::BM == 16 && Base::WM == 1 && Base::NR == 1 && Base::BK % 128 == 0);
    static constexpr index_t WaveBytes = 16 * Base::BK;
};
template <class Base>
struct GroupedInt8NtTilePolicy<GroupedInt8NtWavePrivateProblem<Base>>
    : GroupedInt8NtTilePolicy<Base>
{
    using P                               = GroupedInt8NtWavePrivateProblem<Base>;
    using Parent                          = GroupedInt8NtTilePolicy<Base>;
    static constexpr index_t OperandBytes = P::WN * P::WaveBytes, OperandAllocation = OperandBytes;
    static constexpr index_t SmemBytes =
        OperandBytes > Parent::CStripeBytes ? OperandBytes : Parent::CStripeBytes;
    CK_TILE_HOST_DEVICE static constexpr index_t Offset(index_t row, index_t k)
    {
        return row * P::BK + (k ^ ((row & 7) * 16));
    }
};
template <class Base, class Policy>
struct GroupedInt8NtTilePipeline<GroupedInt8NtWavePrivateProblem<Base>, Policy>
{
    using P                         = GroupedInt8NtWavePrivateProblem<Base>;
    using Mmac                      = typename P::Mmac;
    using Copy                      = typename P::CopyVector;
    using AVec                      = typename Mmac::AVecType;
    using BVec                      = typename Mmac::BVecType;
    using Acc                       = typename Mmac::CVecType;
    static constexpr index_t Copies = P::WaveBytes / (64 * 16);
    struct GlobalTile
    {
        Copy b[Copies];
        AVec a[P::BK / 32];
    };
    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   index_t kb) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        GlobalTile v{};
        const index_t arow = mb + lane % 16;
        static_for<0, P::BK / 32, 1>{}([&](auto kk) {
            constexpr int k = decltype(kk)::value;
            if(arow < arg.M)
                v.a[k] =
                    *reinterpret_cast<const AVec*>(static_cast<const int8_t*>(arg.as_ptr[0]) +
                                                   static_cast<long long>(arow) * arg.stride_As[0] +
                                                   kb + k * 32 + (lane / 16) * 8);
        });
        static_for<0, Copies, 1>{}([&](auto ii) {
            constexpr int i   = decltype(ii)::value;
            const index_t vec = lane + i * 64, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            v.b[i]           = *reinterpret_cast<const Copy*>(
                static_cast<const int8_t*>(arg.bs_ptr[0]) +
                static_cast<long long>(nb + wave * 16 + row) * arg.stride_Bs[0] + kb + ko);
        });
        return v;
    }
    CK_TILE_DEVICE void Stage(const GlobalTile& v, int8_t* smem) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        static_for<0, Copies, 1>{}([&](auto ii) {
            constexpr int i   = decltype(ii)::value;
            const index_t vec = lane + i * 64, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            *reinterpret_cast<Copy*>(smem + wave * P::WaveBytes + Policy::Offset(row, ko)) = v.b[i];
        });
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    }
    CK_TILE_DEVICE void
    Compute(const GlobalTile& v, const int8_t* smem, Acc (&c)[P::MR][P::NR]) const
    {
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        static_for<0, P::BK / 32, 1>{}([&](auto kk) {
            constexpr int k = decltype(kk)::value;
            const auto b    = *reinterpret_cast<const BVec*>(
                smem + wave * P::WaveBytes + Policy::Offset(lane % 16, k * 32 + (lane / 16) * 8));
            Mmac{}(c[0][0], v.a[k], b);
        });
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   int8_t* smem,
                                   Acc (&c)[P::MR][P::NR]) const
    {
        auto current = Load(arg, mb, nb, 0);
        Stage(current, smem);
        for(index_t kb = 0; kb + P::BK < arg.K; kb += P::BK)
        {
            auto next = Load(arg, mb, nb, kb + P::BK);
            __builtin_amdgcn_sched_barrier(0);
            Compute(current, smem, c);
            __builtin_amdgcn_sched_barrier(0);
            // All reads from this wave's private slab retire before it overwrites it.
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            Stage(next, smem);
            current = next;
        }
        Compute(current, smem, c);
        // The existing C epilogue retains its CTA barriers and shares this allocation.
    }
};
} // namespace ck_tile
