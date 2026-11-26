// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_policy.hpp"
namespace ck_tile {
template <class P, class Policy = GroupedInt8NtTilePolicy<P>>
struct GroupedInt8NtTilePipeline
{
    using Mmac = typename P::Mmac;
    using Copy = typename P::CopyVector;
    using Acc  = typename Mmac::CVecType;
    using Rsrc = int32_t __attribute__((ext_vector_type(4)));
    struct GlobalTile
    {
        Copy a[Policy::AVectors], b[Policy::BVectors];
    };
    union alignas(16) OperandPair
    {
        Copy pair;
        typename Mmac::AVecType frag[2];
    };
    static_assert(sizeof(OperandPair) == P::InputVectorBytes);

    static constexpr index_t VecsPerRow  = P::BK / P::InputVectorBytes;
    static constexpr index_t RowsPerStep = P::Threads / VecsPerRow;
    static constexpr index_t ActiveALoads =
        (P::BM * VecsPerRow + P::Threads - 1) / P::Threads;
    static constexpr index_t ActiveBLoads =
        (P::BN * VecsPerRow + P::Threads - 1) / P::Threads;
    static constexpr index_t TotalFastLoads = ActiveALoads + ActiveBLoads;
    static constexpr index_t SwzPitch       = P::BK;
    static constexpr index_t SwzBOffset     = P::BM * SwzPitch;
    static constexpr bool FastXorSwizzle =
        (P::BK >= 64) && (P::BK % 64 == 0) &&
        ((RowsPerStep % 8 == 0) || (RowsPerStep == 4));

    CK_TILE_DEVICE static Rsrc make_rsrc(const void* ptr, uint32_t size_bytes)
    {
        Rsrc r;
        const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
        r[0] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(addr & 0xffffffffu));
        r[1] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(addr >> 32));
        r[2] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(size_bytes));
        r[3] = __builtin_amdgcn_readfirstlane(0x00020000);
        return r;
    }

    CK_TILE_DEVICE static Copy buffer_load_b128(Rsrc rsrc, uint32_t voff, uint32_t soff)
    {
        Copy out;
        asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 offen"
                     : "=v"(out)
                     : "v"(voff), "s"(rsrc), "s"(soff)
                     : "memory");
        return out;
    }

    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,
                                   index_t mb,
                                   index_t nb,
                                   index_t kb) const
    {
        GlobalTile v{};
        const index_t tid = get_thread_id();
        static_for<0, Policy::AVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / P::InputVectorBytes),
                          ko = (vec % (P::BK / P::InputVectorBytes)) * P::InputVectorBytes;
            if(row < P::BM &&
               (P::Aligned || (mb + row < arg.M && kb + ko + P::InputVectorBytes <= arg.K)))
                v.a[j] = *reinterpret_cast<const Copy*>(
                    static_cast<const int8_t*>(arg.as_ptr[0]) +
                    static_cast<long long>(mb + row) * arg.stride_As[0] + kb + ko);
        });
        static_for<0, Policy::BVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / P::InputVectorBytes),
                          ko = (vec % (P::BK / P::InputVectorBytes)) * P::InputVectorBytes;
            if(row < P::BN &&
               (P::Aligned || (nb + row < arg.N && kb + ko + P::InputVectorBytes <= arg.K)))
                v.b[j] = *reinterpret_cast<const Copy*>(
                    static_cast<const int8_t*>(arg.bs_ptr[0]) +
                    static_cast<long long>(nb + row) * arg.stride_Bs[0] + kb + ko);
        });
        return v;
    }

    CK_TILE_DEVICE void Stage(const GlobalTile& v, int8_t* smem) const
    {
        const index_t tid = get_thread_id();
        static_for<0, Policy::AVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / P::InputVectorBytes),
                          ko = (vec % (P::BK / P::InputVectorBytes)) * P::InputVectorBytes;
            if(row < P::BM)
            {
                *reinterpret_cast<Copy*>(smem + row * Policy::OperandPitch + ko) = v.a[j];
            }
        });
        static_for<0, Policy::BVectors, 1>{}([&](auto i) {
            constexpr index_t j = decltype(i)::value;
            const index_t vec = tid + j * P::Threads, row = vec / (P::BK / P::InputVectorBytes),
                          ko = (vec % (P::BK / P::InputVectorBytes)) * P::InputVectorBytes;
            if(row < P::BN)
            {
                *reinterpret_cast<Copy*>(smem + Policy::BPitchOffset + row * Policy::OperandPitch +
                                         ko) = v.b[j];
            }
        });
    }

    CK_TILE_DEVICE void StageFast(const GlobalTile& v, int8_t* smem) const
    {
        const uint32_t tid   = static_cast<uint32_t>(get_thread_id());
        const uint32_t row0  = tid / VecsPerRow;
        const uint32_t col16 = tid % VecsPerRow;
        if constexpr(FastXorSwizzle && (RowsPerStep % 8 == 0))
        {
            const uint32_t ko_swz = (col16 ^ (row0 & 7u)) * P::InputVectorBytes;
            int8_t* const sbase   = smem + row0 * SwzPitch + ko_swz;
            static_for<0, ActiveALoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = TotalFastLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                *reinterpret_cast<Copy*>(sbase + j * RowsPerStep * SwzPitch) = v.a[j];
            });
            static_for<0, ActiveBLoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = ActiveBLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                *reinterpret_cast<Copy*>(sbase + SwzBOffset + j * RowsPerStep * SwzPitch) = v.b[j];
            });
        }
        else if constexpr(FastXorSwizzle && (RowsPerStep == 4))
        {
            int8_t* const sbase0 =
                smem + row0 * SwzPitch + ((col16 ^ (row0 & 7u)) * P::InputVectorBytes);
            int8_t* const sbase1 =
                smem + row0 * SwzPitch + ((col16 ^ ((row0 + 4u) & 7u)) * P::InputVectorBytes);
            static_for<0, ActiveALoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = TotalFastLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                int8_t* const sb = (j & 1) ? sbase1 : sbase0;
                *reinterpret_cast<Copy*>(sb + j * RowsPerStep * SwzPitch) = v.a[j];
            });
            static_for<0, ActiveBLoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = ActiveBLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                int8_t* const sb = (j & 1) ? sbase1 : sbase0;
                *reinterpret_cast<Copy*>(sb + SwzBOffset + j * RowsPerStep * SwzPitch) = v.b[j];
            });
        }
        else
        {
            const uint32_t ko   = col16 * P::InputVectorBytes;
            int8_t* const sbase = smem + row0 * Policy::OperandPitch + ko;
            static_for<0, ActiveALoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = TotalFastLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                *reinterpret_cast<Copy*>(sbase + j * RowsPerStep * Policy::OperandPitch) = v.a[j];
            });
            static_for<0, ActiveBLoads, 1>{}([&](auto i) {
                constexpr index_t j   = decltype(i)::value;
                constexpr index_t rem = ActiveBLoads - 1 - j;
                if constexpr(rem <= 15)
                    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(rem) : "memory");
                else
                    asm volatile("s_waitcnt vmcnt(15)" ::: "memory");
                *reinterpret_cast<Copy*>(sbase + Policy::BPitchOffset +
                                         j * RowsPerStep * Policy::OperandPitch) = v.b[j];
            });
        }
    }

    CK_TILE_DEVICE void ComputeFast(const int8_t* smem, Acc (&c)[P::MR][P::NR]) const
    {
        if constexpr(!FastXorSwizzle)
        {
            Compute(smem, c);
            return;
        }
        const uint32_t tid         = static_cast<uint32_t>(get_thread_id());
        const uint32_t lane        = tid % P::WaveSize;
        const uint32_t wave        = tid / P::WaveSize;
        const uint32_t wm          = wave / P::WN;
        const uint32_t wn          = wave % P::WN;
        const uint32_t r7          = lane & 7u;
        const uint32_t q           = lane / P::MmacRows;
        const uint32_t ko0         = (q ^ r7) * P::InputVectorBytes;
        const uint32_t ko1         = ((q + 4u) ^ r7) * P::InputVectorBytes;
        const uint32_t a_row0      = wm * (P::BM / P::WM) + (lane % P::MmacRows);
        const uint32_t b_row0      = wn * (P::BN / P::WN) + (lane % P::MmacCols);
        const int8_t* const a_sm0  = smem + a_row0 * SwzPitch + ko0;
        const int8_t* const a_sm1  = smem + a_row0 * SwzPitch + ko1;
        const int8_t* const b_sm0  = smem + SwzBOffset + b_row0 * SwzPitch + ko0;
        const int8_t* const b_sm1  = smem + SwzBOffset + b_row0 * SwzPitch + ko1;
        static_for<0, P::BK, 2 * P::MmacK>{}([&](auto kk) {
            constexpr index_t k_base   = decltype(kk)::value;
            constexpr index_t k_high   = k_base & ~index_t(127);
            constexpr bool odd_half    = ((k_base / 64) & 1) != 0;
            const int8_t* const a_smem = odd_half ? a_sm1 : a_sm0;
            const int8_t* const b_smem = odd_half ? b_sm1 : b_sm0;
            OperandPair av[P::MR];
            OperandPair bv[P::NR];
            static_for<0, P::MR, 1>{}([&](auto i) {
                constexpr index_t mi = decltype(i)::value;
                av[mi].pair          = *reinterpret_cast<const Copy*>(
                    a_smem + (mi * P::MmacRows * SwzPitch + k_high));
            });
            static_for<0, P::NR, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(j)::value;
                bv[ni].pair          = *reinterpret_cast<const Copy*>(
                    b_smem + (ni * P::MmacCols * SwzPitch + k_high));
            });
            static_for<0, 2, 1>{}([&](auto s) {
                constexpr index_t sub = decltype(s)::value;
                static_for<0, P::MR, 1>{}([&](auto i) {
                    static_for<0, P::NR, 1>{}([&](auto j) {
                        Mmac{}(c[decltype(i)::value][decltype(j)::value],
                               av[decltype(i)::value].frag[sub],
                               bv[decltype(j)::value].frag[sub]);
                    });
                });
            });
        });
    }

    CK_TILE_DEVICE void Compute(const int8_t* smem, Acc (&c)[P::MR][P::NR]) const
    {
        const index_t lane = get_thread_id() % P::WaveSize, wave = get_thread_id() / P::WaveSize;
        const index_t wm = wave / P::WN, wn = wave % P::WN,
                      ko = (lane / P::MmacRows) * P::InputVectorBytes;
        const uint32_t a_row0      = static_cast<uint32_t>(wm * (P::BM / P::WM) + lane % P::MmacRows);
        const uint32_t b_row0      = static_cast<uint32_t>(wn * (P::BN / P::WN) + lane % P::MmacCols);
        const int8_t* const a_smem = smem + a_row0 * Policy::OperandPitch + ko;
        const int8_t* const b_smem = smem + Policy::BPitchOffset + b_row0 * Policy::OperandPitch + ko;
        static_for<0, P::BK, 2 * P::MmacK>{}([&](auto kk) {
            constexpr index_t k_base = decltype(kk)::value;
            OperandPair av[P::MR];
            OperandPair bv[P::NR];
            static_for<0, P::MR, 1>{}([&](auto i) {
                constexpr index_t mi = decltype(i)::value;
                av[mi].pair          = *reinterpret_cast<const Copy*>(
                    a_smem + (mi * P::MmacRows * Policy::OperandPitch + k_base));
            });
            static_for<0, P::NR, 1>{}([&](auto j) {
                constexpr index_t ni = decltype(j)::value;
                bv[ni].pair          = *reinterpret_cast<const Copy*>(
                    b_smem + (ni * P::MmacCols * Policy::OperandPitch + k_base));
            });
            static_for<0, 2, 1>{}([&](auto s) {
                constexpr index_t sub = decltype(s)::value;
                static_for<0, P::MR, 1>{}([&](auto i) {
                    static_for<0, P::NR, 1>{}([&](auto j) {
                        Mmac{}(c[decltype(i)::value][decltype(j)::value],
                               av[decltype(i)::value].frag[sub],
                               bv[decltype(j)::value].frag[sub]);
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
        if constexpr(P::Peeled && P::Aligned && (P::Threads % VecsPerRow == 0) &&
                     (P::BM % RowsPerStep == 0) && (P::BN % RowsPerStep == 0) &&
                     (TotalFastLoads <= 16))
        {
            const long long a_bytes = static_cast<long long>(P::BM) * arg.stride_As[0];
            const long long b_bytes = static_cast<long long>(P::BN) * arg.stride_Bs[0];
            if(a_bytes <= 0x7fffffffLL && b_bytes <= 0x7fffffffLL)
            {
                const uint32_t tid  = static_cast<uint32_t>(get_thread_id());
                const uint32_t row0 = tid / VecsPerRow;
                const uint32_t ko   = (tid % VecsPerRow) * P::InputVectorBytes;
                const uint32_t as   = static_cast<uint32_t>(arg.stride_As[0]);
                const uint32_t bs   = static_cast<uint32_t>(arg.stride_Bs[0]);
                const Rsrc a_rsrc   = make_rsrc(static_cast<const int8_t*>(arg.as_ptr[0]) +
                                                  static_cast<long long>(mb) * arg.stride_As[0],
                                              static_cast<uint32_t>(a_bytes));
                const Rsrc b_rsrc   = make_rsrc(static_cast<const int8_t*>(arg.bs_ptr[0]) +
                                                  static_cast<long long>(nb) * arg.stride_Bs[0],
                                              static_cast<uint32_t>(b_bytes));
                uint32_t a_off[ActiveALoads];
                uint32_t b_off[ActiveBLoads];
                static_for<0, ActiveALoads, 1>{}([&](auto i) {
                    constexpr index_t j = decltype(i)::value;
                    a_off[j]            = (row0 + j * RowsPerStep) * as + ko;
                });
                static_for<0, ActiveBLoads, 1>{}([&](auto i) {
                    constexpr index_t j = decltype(i)::value;
                    b_off[j]            = (row0 + j * RowsPerStep) * bs + ko;
                });
                auto load_fast = [&](index_t kb_offset) {
                    GlobalTile v{};
                    const uint32_t soff =
                        __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(kb_offset));
                    static_for<0, ActiveALoads, 1>{}([&](auto i) {
                        constexpr index_t j = decltype(i)::value;
                        v.a[j]              = buffer_load_b128(a_rsrc, a_off[j], soff);
                    });
                    static_for<0, ActiveBLoads, 1>{}([&](auto i) {
                        constexpr index_t j = decltype(i)::value;
                        v.b[j]              = buffer_load_b128(b_rsrc, b_off[j], soff);
                    });
                    return v;
                };
                auto current = load_fast(0);
                StageFast(current, smem);
                block_sync_lds();
                for(index_t kb = 0; kb + P::BK < arg.K; kb += P::BK)
                {
                    auto next = load_fast(kb + P::BK);
                    __builtin_amdgcn_sched_barrier(0);
                    ComputeFast(smem, c);
                    __builtin_amdgcn_sched_barrier(0);
                    block_sync_lds();
                    StageFast(next, smem);
                    block_sync_lds();
                }
                ComputeFast(smem, c);
                return;
            }
        }
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
                Stage(next, next_smem_alias(smem));
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

    CK_TILE_DEVICE static int8_t* next_smem_alias(int8_t* smem) { return smem; }
};
} // namespace ck_tile
