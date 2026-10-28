// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtStreamPolicy
{
    static constexpr index_t InputBytes=(P::BM+P::BN)*P::BK;
    static constexpr index_t StageThreads=P::ProducerWaves?P::ProducerWaves*64:P::ComputeThreads;
    static constexpr index_t Vectors=(P::BM+P::BN)*(P::BK/16),Issues=Vectors/StageThreads;
    static constexpr index_t Pending=(P::Stages-2)*Issues;
    // Reserve64KiB for the generic K64 ping-pong fallback as well.
    static constexpr index_t SmemBytes=65536;
    static_assert(Vectors%StageThreads==0 && InputBytes*P::Stages<=SmemBytes);
};
}
