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
    using Mmac                                = WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using CopyVector                          = ext_vector_t<int32_t, 4>;
    static constexpr index_t WaveSize         = 64;
    static constexpr index_t MmacRows         = Mmac::kM;
    static constexpr index_t MmacCols         = Mmac::kN;
    static constexpr index_t MmacK            = Mmac::kK;
    static constexpr index_t MmacInputBytes   = sizeof(typename Mmac::AVecType);
    static constexpr index_t InputVectorBytes = sizeof(CopyVector);
    static constexpr index_t AccElements      = vector_traits<typename Mmac::CVecType>::vector_size;
    static constexpr index_t BM = BM_, BN = BN_, BK = BK_, WM = WM_, WN = WN_;
    static constexpr bool MFirst     = MFirst_;
    static constexpr bool Aligned    = Aligned_;
    static constexpr bool Peeled     = Peeled_;
    static constexpr bool BatchedEpi = BatchedEpi_, CopyArg = CopyArg_;
    static constexpr index_t MinBlocks = MinBlocks_, MGroup = MGroup_;
    static_assert(BK == 64 || BK == 128 || BK == 256 || BK == 512,
                  "Supported packed K tiles: 64/128/256/512");
    static constexpr index_t Threads = WM * WN * WaveSize, MR = BM / (WM * MmacRows),
                             NR = BN / (WN * MmacCols);
    static_assert(BM % (WM * MmacRows) == 0 && BN % (WN * MmacCols) == 0 && BK % MmacK == 0);
};
} // namespace ck_tile
