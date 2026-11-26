// SPDX-License-Identifier: MIT
#pragma once

#include "../kernels/grouped_int8_nt_wasp_problem.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

namespace ck_tile::int8_nt_instances {

template <class T, bool DirectEpilogue = false>
struct TypeTag
{
    using type                              = T;
    static constexpr bool UseDirectEpilogue = DirectEpilogue;
};

// -----------------------------------------------------------------------------
// 1. Cached & Scaled Route Problem Instances
// -----------------------------------------------------------------------------
using RawAlignedM16N32K512 = GroupedInt8NtTileProblem<
    /* BlockM = */ 16,
    /* BlockN = */ 32,
    /* BlockK = */ 512,
    /* WavesM = */ 1,
    /* WavesN = */ 2,
    /* MFirst = */ false,
    /* Aligned = */ true,
    /* PeelKLoop = */ true,
    /* BatchedEpilogue = */ true,
    /* MinBlocks = */ 2,
    /* CopyArgs = */ true>;
using RawM32N64K256W4 = GroupedInt8NtTileProblem<
    /* BlockM = */ 32,
    /* BlockN = */ 64,
    /* BlockK = */ 256,
    /* WavesM = */ 2,
    /* WavesN = */ 2,
    /* MFirst = */ false,
    /* Aligned = */ true,
    /* PeelKLoop = */ true,
    /* BatchedEpilogue = */ true,
    /* MinBlocks = */ 2,
    /* CopyArgs = */ true>;
// Stable route handles identify precompiled geometry/policy, never workload IDs.
using RawTransposeM256N256 = GroupedInt8NtWaspCachedProblem<
    /* Transpose = */ true,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 4,
    /* AReadBatch = */ 4,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset>;
using RawTransposeM256N256Record1024 = GroupedInt8NtWaspCachedProblem<
    /* Transpose = */ true,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 4,
    /* AReadBatch = */ 4,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 1024,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset>;
// Original NT weights with byte-offset addressing and cache swizzle.
using RawM256N256 = GroupedInt8NtWaspCachedProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 4,
    /* AReadBatch = */ 4,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset>;

using RawM128N256 = GroupedInt8NtWaspSplitMProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 4,
    /* AReadBatch = */ 2,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 8>;

using RawM128N128 = GroupedInt8NtWaspSplitMProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 4,
    /* MGroup = */ 1,
    /* AReadBatch = */ 4,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 4>;

using RawM64N256 = GroupedInt8NtWaspSmallMProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 1,
    /* AReadBatch = */ 2,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 8>;

using RawM64N64 = GroupedInt8NtWaspSmallMProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 2,
    /* MGroup = */ 1,
    /* AReadBatch = */ 2,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 2>;

using RawM32N256 = GroupedInt8NtWaspM32Problem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 8,
    /* MGroup = */ 1,
    /* AReadBatch = */ 2,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 8>;

using RawM32N64 = GroupedInt8NtWaspM32Problem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 2,
    /* MGroup = */ 1,
    /* AReadBatch = */ 1,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 2>;

using RawM256N384 = GroupedInt8NtWaspPublicWideNProblem<
    /* Transpose = */ false,
    /* ConsumerSplit = */ 12,
    /* MGroup = */ 4,
    /* AReadBatch = */ 2,
    /* APrefetch = */ true,
    /* OneBarrier = */ true,
    /* RecordCap = */ 0,
    /* EarlyB = */ false,
    /* BAddress = */ int8_nt_b_address::ByteOffset,
    /* ConsumerWaves = */ 12>;

template <bool BFloat>
using RawPairedM256N384Problem =
    std::conditional_t<BFloat,
                       GroupedInt8NtWaspPublicWideNProblem<
                           /* Transpose = */ false,
                           /* ConsumerSplit = */ 12,
                           /* MGroup = */ 1,
                           /* AReadBatch = */ 2,
                           /* APrefetch = */ true,
                           /* OneBarrier = */ true,
                           /* RecordCap = */ 0,
                           /* EarlyB = */ false,
                           /* BAddress = */ int8_nt_b_address::ByteOffset,
                           /* ConsumerWaves = */ 12>,
                       RawM256N384>;

using ScaledRawM256N384MaskedM4 = GroupedInt8NtWaspRawMaskedPrefetchProblem<4, false, false>;
using ScaledRawM256N384MaskedM8 = GroupedInt8NtWaspRawMaskedPrefetchProblem<8, true, true>;
using ScaledRawM256N384MaskedM16 = GroupedInt8NtWaspRawMaskedPrefetchProblem<8, false, false, true>;

// -----------------------------------------------------------------------------
// 2. Shared tile instance definitions
// -----------------------------------------------------------------------------
using DefaultFallbackTileProblem = GroupedInt8NtTileProblem<
    /* BlockM = */ 16,
    /* BlockN = */ 64,
    /* BlockK = */ 128,
    /* WavesM = */ 1,
    /* WavesN = */ 2,
    /* MFirst = */ false,
    /* Aligned = */ false,
    /* PeelKLoop = */ true,
    /* BatchedEpilogue = */ true,
    /* MinBlocks = */ 2,
    /* CopyArgs = */ true>;
using DirectSmallMTileProblem = GroupedInt8NtTileProblem<
    /* BlockM = */ 16,
    /* BlockN = */ 32,
    /* BlockK = */ 512,
    /* WavesM = */ 1,
    /* WavesN = */ 2,
    /* MFirst = */ false,
    /* Aligned = */ false,
    /* PeelKLoop = */ true,
    /* BatchedEpilogue = */ true,
    /* MinBlocks = */ 2,
    /* CopyArgs = */ true>;

template <class OnTile, class OnWasp, class OnDirectM16, class OnDefault>
int dispatch_grouped_int8_nt_instance(GroupedGemmInstanceId id,
                                      OnTile&& on_tile,
                                      OnWasp&& on_wasp,
                                      OnDirectM16&& on_direct_m16,
                                      OnDefault&& on_default)
{
    using Instance = GroupedGemmInstanceId;
    switch(id)
    {
#define CK_INT8_LEGACY_TILE(id, ...) \
    case Instance::id: return on_tile(TypeTag<__VA_ARGS__>{});
#define CK_INT8_LEGACY_WASP(id, ...) \
    case Instance::id: return on_wasp(TypeTag<__VA_ARGS__>{});
#define CK_INT8_LEGACY_DIRECT(id) \
    case Instance::id: return on_direct_m16();
#define CK_INT8_LEGACY_DIRECT_EPI(id, ...) \
    case Instance::id: return on_tile(TypeTag<__VA_ARGS__, true>{});
#include "grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
    default: return on_default();
    }
}

} // namespace ck_tile::int8_nt_instances
