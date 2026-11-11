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
    case Instance::int8_nt_direct_m16:
        return on_direct_m16();
    case Instance::int8_nt_m32:
        return on_tile(TypeTag<GroupedInt8NtTileProblem<32, 128>>{});
    case Instance::int8_nt_m64:
        return on_tile(TypeTag<GroupedInt8NtTileProblem<64, 128>>{});
    case Instance::int8_nt_m128:
        return on_tile(TypeTag<GroupedInt8NtTileProblem<128, 128>>{});
    case Instance::int8_nt_m128_n256:
        return on_tile(TypeTag<GroupedInt8NtTileProblem<128, 256, 64, 2, 4>>{});
    case Instance::int8_nt_m128_aligned:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false>>{});
    case Instance::int8_nt_m256_n256_aligned:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true>>{});
    case Instance::int8_nt_m128_n256_aligned:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<128, 256, 64, 2, 4, true, true, true, false, 2, true>>{});
    case Instance::int8_nt_m256_n256_padded:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, false, true, false, 1, true>>{});
    case Instance::int8_nt_m128_padded:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true>>{});
    case Instance::int8_nt_m256_n256_ngroup:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true, 1>>{});
    case Instance::int8_nt_m128_mgroup8:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false, 8>>{});
    case Instance::int8_nt_m128_padded_mgroup8:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true, 8>>{});
    case Instance::int8_nt_m16_n256_aligned:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<16, 256, 64, 1, 4, false, true, true, true, 1, true>>{});
    case Instance::int8_nt_small_m_n64_k128:
        return on_tile(TypeTag<DefaultFallbackTileProblem>{});
    case Instance::int8_nt_small_m_n64_k256:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<16, 64, 256, 1, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_small_m_n128_k256:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<16, 128, 256, 1, 4, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_small_g_n32_k512:
        return on_tile(TypeTag<DirectSmallMTileProblem>{});
    case Instance::int8_nt_m32_n64_k128:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<32, 64, 128, 1, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_m32_n64_k256:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<32, 64, 256, 1, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_m32_n128_k256:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<32, 128, 256, 2, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_m64_n128_k128:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<64, 128, 128, 2, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_m64_n128_k256:
        return on_tile(TypeTag<
            GroupedInt8NtTileProblem<64, 128, 256, 2, 2, false, false, true, true, 1, true>>{});
    case Instance::int8_nt_small_m_n32_k512_direct:
        return on_direct_epi(TypeTag<DirectSmallMTileProblem>{});
    case Instance::int8_nt_wasp_a4_transposed:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<256, true, 8, 4, 4, true, true, 0, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_a8_transposed:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<256, true, 8, 4, 8, true, true, 0, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_a4:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<256, false, 8, 4, 4, true, true, 0, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_record1024_a4:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<256, false, 8, 4, 4, true, true, 1024, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_record1024_a4_transposed:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<256, true, 8, 4, 4, true, true, 1024, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_m128_a2:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<128, false, 8, 1, 2, true, true, 0, false, false, 0, 8, true>>{});
    case Instance::int8_nt_wasp_m128_record1024_a2:
        return on_wasp(
            TypeTag<GroupedInt8NtWaspCoreProblem<128, false, 8, 1, 2, true, true, 1024, false, false, 0, 8, true>>{});
    default:
        return on_default();
    }
}

} // namespace ck_tile::int8_nt_instances
