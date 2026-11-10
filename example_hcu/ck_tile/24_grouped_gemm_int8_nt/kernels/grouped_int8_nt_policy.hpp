// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template <class P>
struct GroupedInt8NtTilePolicy
{
    // 16 bytes of row padding separate the banks used by adjacent MMAC rows.
    // Native packed vectors keep each int8 fragment in two dword registers.
    static constexpr index_t OperandPitch      = P::BK + 16;
    static constexpr index_t BPitchOffset      = P::BM * OperandPitch;
    static constexpr index_t CStripeRows       = P::BatchedEpi ? P::WM * 16 : 16,
                             CPitch            = P::BatchedEpi ? P::BN : P::BN + 4;
    static constexpr index_t OperandBytes      = (P::BM + P::BN) * OperandPitch;
    static constexpr index_t CStripeBytes      = CStripeRows * CPitch * 4;
    static constexpr index_t OperandAllocation = OperandBytes;
    static constexpr index_t SmemBytes =
        OperandAllocation > CStripeBytes ? OperandAllocation : CStripeBytes;
    static constexpr index_t AVectors = (P::BM * P::BK / 16 + P::Threads - 1) / P::Threads;
    static constexpr index_t BVectors = (P::BN * P::BK / 16 + P::Threads - 1) / P::Threads;
};
} // namespace ck_tile
