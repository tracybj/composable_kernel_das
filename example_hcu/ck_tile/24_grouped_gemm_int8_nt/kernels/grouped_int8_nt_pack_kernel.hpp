// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_layout.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "../include/grouped_gemm_int8_nt_device_args.h"
#include <cstdint>
namespace ck_tile {
struct Int8CachedWeightPackProblem : Int8NtPackedWeightLayout
{
    static constexpr index_t Threads = 256;
    static constexpr index_t NSteps  = 4;
    static constexpr index_t KSteps  = 2;
};
using Int8CachedWeightPackArgs = ck_tile_hcu_grouped_gemm_int8_pack_arg_v2;
template <class Problem = Int8CachedWeightPackProblem>
struct Int8CachedWeightPackKernel
{
    CK_TILE_DEVICE void operator()(const Int8CachedWeightPackArgs* args) const
    {
        using Layout = Int8NtPackedWeightLayout;
        static_assert(Problem::NTile == Layout::NTile && Problem::KTile == Layout::KTile &&
                      Problem::VectorBytes == Layout::VectorBytes);
        const auto a           = args[blockIdx.z];
        const uint32_t lane    = get_thread_id() % Problem::WaveSize,
                       wave    = get_thread_id() / Problem::WaveSize;
        const uint64_t waves   = Problem::Threads / Problem::WaveSize;
        const uint64_t n_tiles = static_cast<uint64_t>(a.n / Problem::NTile),
                       k_tiles = static_cast<uint64_t>(a.k / Problem::KTile);
        const uint64_t nt_base = static_cast<uint64_t>(blockIdx.x) * waves * Problem::NSteps + wave,
                       kt_base = static_cast<uint64_t>(blockIdx.y) * Problem::KSteps;
        const uint64_t ni      = lane % Problem::NTile, kg = lane / Problem::NTile;
        using Vec =
            int32_t __attribute__((ext_vector_type(Problem::VectorBytes / sizeof(int32_t))));
        static_for<0, Problem::KSteps, 1>{}([&](auto ki) {
            const uint64_t kt = kt_base + decltype(ki)::value;
            if(kt >= k_tiles)
                return;
            static_for<0, Problem::NSteps, 1>{}([&](auto step) {
                const uint64_t nt = nt_base + decltype(step)::value * waves;
                if(nt >= n_tiles)
                    return;
                const uint64_t packed_offset =
                    Layout::vector_offset(kt, n_tiles, nt, uint64_t(lane));
                const auto* source = a.source + (nt * Problem::NTile + ni) * a.stride +
                                     kt * Problem::KTile + kg * Problem::VectorBytes;
                *reinterpret_cast<Vec*>(a.packed + packed_offset) =
                    *reinterpret_cast<const Vec*>(source);
            });
        });
    }
};
} // namespace ck_tile
