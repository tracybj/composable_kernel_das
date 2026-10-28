// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P>
struct GroupedInt8NtD2LGeometryPolicy
{
    static constexpr index_t InputBytes=(P::BM+P::BN)*P::BK;
    static constexpr index_t InputAllocation=InputBytes*(P::DoubleBuffer?2:1);
    static constexpr index_t CStripeRows=P::BatchedEpi?P::WM*16:16;
    static constexpr index_t CPitch=P::BatchedEpi?P::BN:P::BN+4;
    static constexpr index_t OutputBytes=CStripeRows*CPitch*4;
    static constexpr index_t SmemBytes=InputAllocation>OutputBytes?InputAllocation:OutputBytes;
    static constexpr index_t StageThreads=P::ProducerWaves?P::ProducerWaves*64:P::ComputeThreads;
    static constexpr index_t Vectors=(P::BM+P::BN)*4;
    static constexpr index_t Issues=(Vectors+StageThreads-1)/StageThreads;
    static_assert(SmemBytes<=65536);
};
}
