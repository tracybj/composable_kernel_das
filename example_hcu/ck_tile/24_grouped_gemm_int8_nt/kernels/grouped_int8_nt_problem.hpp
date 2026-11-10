// SPDX-License-Identifier: MIT
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac_impl.hpp"
#include <type_traits>
namespace ck_tile {
template <index_t BM_,
          index_t BN_,
          index_t BK_        = 64,
          index_t WM_        = 2,
          index_t WN_        = 2,
          bool MFirst_       = false,
          bool Aligned_      = false,
          bool Peeled_       = false,
          bool BatchedEpi_   = false,
          index_t MinBlocks_ = 1,
          bool CopyArg_      = false,
          index_t MGroup_    = 0>
struct GroupedInt8NtTileProblem
{
    using Mmac                  = WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using CopyVector            = ext_vector_t<int32_t, 4>;
    static constexpr index_t BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
    static constexpr bool MFirst     = MFirst_;
    static constexpr bool Aligned    = Aligned_;
    static constexpr bool Peeled     = Peeled_;
    static constexpr bool BatchedEpi = BatchedEpi_, CopyArg = CopyArg_;
    static constexpr index_t MinBlocks = MinBlocks_, MGroup = MGroup_;
    static_assert(BK == 64 || BK == 128 || BK == 256 || BK == 512,
                  "Supported packed K tiles: 64/128/256/512");
    static constexpr index_t Threads = WM * WN * 64, MR = BM / (WM * 16), NR = BN / (WN * 16);
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 16) == 0 && BK % 32 == 0);
};
} // namespace ck_tile
