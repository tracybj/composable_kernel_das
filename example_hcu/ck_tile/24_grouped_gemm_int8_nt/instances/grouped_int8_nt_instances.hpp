// SPDX-License-Identifier: MIT
#pragma once

#include "../kernels/grouped_int8_nt_wasp_problem.hpp"
#include "../kernels/grouped_int8_nt_packed_pipeline.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

namespace ck_tile::int8_nt_instances {

template <class T>
struct TypeTag
{
    using type = T;
};

// -----------------------------------------------------------------------------
// 1. Cached & Scaled Route Problem Instances
// -----------------------------------------------------------------------------
using RawAlignedM16N32K512 =
    GroupedInt8NtTileProblem<16, 32, 512, 1, 2, false, true, true, true, 1, true>;
using RawM32N64K256W4 =
    GroupedInt8NtTileProblem<32, 64, 256, 2, 2, false, false, true, true, 1, true>;
using PackedM16N64K256 = GroupedInt8NtPackedSmallProblem<
    GroupedInt8NtTileProblem<16, 64, 256, 1, 2, false, false, true, true, 1, true>,
    true,
    true>;

// Stable route handles identify precompiled geometry/policy, never workload IDs.
using RawM256N256 = GroupedInt8NtWaspCachedProblem<false, 8, 4, 4, true, true, 0, false, false, 1>;
using RawM256N256Record1024 =
    GroupedInt8NtWaspCachedProblem<false, 8, 4, 4, true, true, 1024, false, false, 1>;
using RawTransposeM256N256 =
    GroupedInt8NtWaspCachedProblem<true, 8, 4, 4, true, true, 0, false, false, 1>;
using RawTransposeM256N256Record1024 =
    GroupedInt8NtWaspCachedProblem<true, 8, 4, 4, true, true, 1024, false, false, 1>;
using PackedM256N256 =
    GroupedInt8NtWaspCachedProblem<false, 8, 4, 4, true, true, 0, false, true, 0>;
using PackedM128N256 =
    GroupedInt8NtWaspSplitMProblem<false, 8, 4, 2, true, true, 0, false, true, 0, 8>;
using PackedM128N128 =
    GroupedInt8NtWaspSplitMProblem<false, 4, 1, 4, true, true, 0, false, true, 0, 4>;
using PackedM64N256 =
    GroupedInt8NtWaspSmallMProblem<false, 8, 1, 2, true, true, 0, false, true, 1, 8>;
using PackedM64N64 =
    GroupedInt8NtWaspSmallMProblem<false, 2, 1, 2, true, true, 0, false, true, 1, 2>;
using PackedM32N256 = GroupedInt8NtWaspM32Problem<false, 8, 1, 2, true, true, 0, false, true, 1, 8>;
using PackedM32N64  = GroupedInt8NtWaspM32Problem<false, 2, 1, 1, true, true, 0, false, true, 1, 2>;
using PackedM256N384 =
    GroupedInt8NtWaspPublicWideNProblem<false, 12, 4, 2, true, true, 0, false, true, 0, 12>;
using PackedM256N384MaskedM4 = GroupedInt8NtWaspMaskedWideProblem<4>;
using PackedM256N384MaskedM8 = GroupedInt8NtWaspMaskedWideProblem<8>;

// Cross-half A prefetch is validated for masked scaled output only. Keep the
// original problem types for INT32 and the unmasked WASP implementations.
using ScaledPackedM256N384MaskedM4 = GroupedInt8NtWaspScaledMaskedPrefetchProblem<4>;
using ScaledPackedM256N384MaskedM8 = GroupedInt8NtWaspScaledMaskedPrefetchProblem<8>;

// -----------------------------------------------------------------------------
// 2. Unified Legacy Instance Table Dispatcher (shared by Raw & Scaled APIs)
// -----------------------------------------------------------------------------
using DefaultFallbackTileProblem =
    GroupedInt8NtTileProblem<16, 64, 128, 1, 2, false, false, true, true, 1, true>;
using DirectSmallMTileProblem =
    GroupedInt8NtTileProblem<16, 32, 512, 1, 2, false, false, true, true, 1, true>;

template <class OnTile, class OnWasp, class OnDirectM16, class OnDirectEpi, class OnDefault>
int dispatch_grouped_int8_nt_instance(GroupedGemmInstanceId id,
                                      OnTile&& on_tile,
                                      OnWasp&& on_wasp,
                                      OnDirectM16&& on_direct_m16,
                                      OnDirectEpi&& on_direct_epi,
                                      OnDefault&& on_default)
{
    using Instance = GroupedGemmInstanceId;
    switch(id)
    {
#define CK_INT8_LEGACY_TILE(id, ...) case Instance::id: return on_tile(TypeTag<__VA_ARGS__>{});
#define CK_INT8_LEGACY_WASP(id, ...) case Instance::id: return on_wasp(TypeTag<__VA_ARGS__>{});
#define CK_INT8_LEGACY_DIRECT(id) case Instance::id: return on_direct_m16();
#define CK_INT8_LEGACY_DIRECT_EPI(id, ...) case Instance::id: return on_direct_epi(TypeTag<__VA_ARGS__>{});
#include "grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
    default:
        return on_default();
    }
}

} // namespace ck_tile::int8_nt_instances
