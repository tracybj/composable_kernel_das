// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtBypassBLifetimePolicy
{
    static constexpr index_t StageThreads=256,AIssues=8,ASlotBytes=P::BM*128,SmemBytes=2*ASlotBytes;
    static_assert(P::BM==256 && P::BN==256 && SmemBytes==65536);
};
}
