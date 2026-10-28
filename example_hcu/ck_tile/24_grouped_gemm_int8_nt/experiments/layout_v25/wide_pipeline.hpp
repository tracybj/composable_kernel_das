// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_policy.hpp"
namespace ck_tile {
template <class P, class Policy = GroupedInt8NtTilePolicy<P>>
struct GroupedInt8NtWideReadPipeline
{
    using Mmac = typename P::Mmac;
    using Copy = typename P::CopyVector;
    using Acc  = typename Mmac::CVecType;
    struct GlobalTile
    {
        Copy a[Policy::AVectors], b[Policy::BVectors];
    };
    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   index_t kb) const
    {
        GlobalTile v{};
        const index_t tid = get_thread_id();
        static_for<0, Policy::AVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            if(row < P::BM && (P::Aligned || (mb + row < arg.M && kb + ko + 16 <= arg.K)))
                v.a[j] = *reinterpret_cast<const Copy*>(
                    static_cast<const int8_t*>(arg.as_ptr[0]) +
                    static_cast<long long>(mb + row) * arg.stride_As[0] + kb + ko);
        });
        static_for<0, Policy::BVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            if(row < P::BN && (P::Aligned || (nb + row < arg.N && kb + ko + 16 <= arg.K)))
                v.b[j] = *reinterpret_cast<const Copy*>(
                    static_cast<const int8_t*>(arg.bs_ptr[0]) +
                    static_cast<long long>(nb + row) * arg.stride_Bs[0] + kb + ko);
        });
        return v;
    }
    CK_TILE_DEVICE void StorePair(int8_t* row, index_t ko, const Copy& value) const
    {
        // Interleave K32 halves so a single aligned b128 read feeds two MMACs.
        const index_t first=(ko%32)/8*16+(ko/32)*8;
        const index_t second=((ko+8)%32)/8*16+((ko+8)/32)*8;
        *reinterpret_cast<typename Mmac::AVecType*>(row+first)=
            __builtin_shufflevector(value,value,0,1);
        *reinterpret_cast<typename Mmac::AVecType*>(row+second)=
            __builtin_shufflevector(value,value,2,3);
    }
    CK_TILE_DEVICE void Stage(const GlobalTile& v, int8_t* smem) const
    {
        const index_t tid = get_thread_id();
        static_for<0, Policy::AVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            if(row < P::BM)
            {
                StorePair(smem + row * Policy::OperandPitch, ko, v.a[j]);
            }
        });
        static_for<0, Policy::BVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / 16),
                          ko = (vec % (P::BK / 16)) * 16;
            if(row < P::BN)
            {
                StorePair(smem + Policy::BPitchOffset + row * Policy::OperandPitch, ko, v.b[j]);
            }
        });
    }
    CK_TILE_DEVICE void Compute(const int8_t* smem, Acc (&c)[P::MR][P::NR]) const
    {
        static_assert(P::BK==64, "Wide-read experiment has paired K32 fragments");
        const index_t lane=get_thread_id()%64, wave=get_thread_id()/64;
        const index_t wm=wave/P::WN, wn=wave%P::WN, ko=(lane/16)*16;
        Copy av[P::MR],bv[P::NR];
        static_for<0,P::MR,1>{}([&](auto i){
            constexpr int mi=decltype(i)::value;
            const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;
            av[mi]=*reinterpret_cast<const Copy*>(smem+row*Policy::OperandPitch+ko);
        });
        static_for<0,P::NR,1>{}([&](auto j){
            constexpr int ni=decltype(j)::value;
            const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;
            bv[ni]=*reinterpret_cast<const Copy*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+ko);
        });
        static_for<0,2,1>{}([&](auto kk){
            constexpr int ki=decltype(kk)::value;
            static_for<0,P::MR,1>{}([&](auto i){
                constexpr int mi=decltype(i)::value;
                auto a=__builtin_shufflevector(av[mi],av[mi],ki*2,ki*2+1);
                static_for<0,P::NR,1>{}([&](auto j){
                    constexpr int ni=decltype(j)::value;
                    auto b=__builtin_shufflevector(bv[ni],bv[ni],ki*2,ki*2+1);
                    Mmac{}(c[mi][ni],a,b);
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
        auto current = Load(arg, mb, nb, 0);
        Stage(current, smem);
        block_sync_lds();
        // The next global tile stays packed in VGPR while MMAC consumes LDS.
        // All waves finish reading before Stage overwrites this single buffer.
        if constexpr(P::Peeled)
        {
            for(index_t kb = 0; kb + P::BK < arg.K; kb += P::BK)
            {
                auto next = Load(arg, mb, nb, kb + P::BK);
                __builtin_amdgcn_sched_barrier(0);
                Compute(smem, c);
                __builtin_amdgcn_sched_barrier(0);
                block_sync_lds();
                Stage(next, smem);
                block_sync_lds();
            }
            Compute(smem, c);
        }
        else
        {
            for(index_t kb = 0; kb < arg.K; kb += P::BK)
            {
                GlobalTile next{};
                if(kb + P::BK < arg.K)
                    next = Load(arg, mb, nb, kb + P::BK);
                int8_t* current_smem = smem;
                Compute(current_smem, c);
                if(kb + P::BK < arg.K)
                {
                    int8_t* next_smem = smem;
                    block_sync_lds();
                    Stage(next, next_smem);
                    block_sync_lds();
                }
            }
        }
    }
};
} // namespace ck_tile
