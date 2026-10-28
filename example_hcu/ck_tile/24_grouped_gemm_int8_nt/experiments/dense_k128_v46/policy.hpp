// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtDenseKPolicy
{
    static constexpr index_t StageThreads=P::ProducerWaves?P::ProducerWaves*64:P::ComputeThreads;
    static constexpr index_t AVectors=P::BM*P::BK/16/StageThreads,BVectors=P::BN*P::BK/16/StageThreads;
    static constexpr index_t BPitchOffset=P::BM*P::BK,SmemBytes=(P::BM+P::BN)*P::BK;
    static_assert(SmemBytes==65536);
};
}
