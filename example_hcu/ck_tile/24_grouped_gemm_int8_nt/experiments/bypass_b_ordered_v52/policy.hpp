// SPDX-License-Identifier: MIT
#pragma once
#include "problem.hpp"
namespace ck_tile {
template<class P> struct GroupedInt8NtBypassBOrderedPolicy
{
    static constexpr index_t StageThreads=256,AIssues=8,ASlotBytes=P::BM*128,SmemBytes=2*ASlotBytes;
    static_assert(P::BM==256 && P::BN==256 && SmemBytes==65536);
};

// Keep the native INT8 fragment ABI while making compute a scheduling boundary.
template<class P> struct GroupedInt8NtBypassBOrderedMmacPolicy
{
    using Native=typename P::Mmac;
    CK_TILE_DEVICE void operator()(typename Native::CVecType& c,
        const typename Native::AVecType& a,const typename Native::BVecType& b) const
    {
        if constexpr(P::LifetimeMode==2)
            asm volatile("v_mmac_i32_16x16x32_i8 %0, %1, %2, %0"
                : "+v"(c) : "v"(a),"v"(b) : "memory");
        else
        {
            Native{}(c,a,b);
            // Tie the resulting fragment to the compiler memory boundary.
            asm volatile("" : "+v"(c) : : "memory");
        }
    }
};
}
