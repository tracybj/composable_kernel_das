// SPDX-License-Identifier: MIT
#pragma once
#include "../dense_k128_v46/problem.hpp"
namespace ck_tile {
template<bool Aligned_=true,bool PrefetchB_=true,int Group_=4,bool Priority_=false,int Shift_=0,int Mode_=0>
struct GroupedInt8NtBypassBLifetimeProblem : GroupedInt8NtDenseKProblem<4,false,true,Aligned_,1,8,Group_,true>
{
    static constexpr bool PrefetchB=PrefetchB_,ConsumerPriority=Priority_;
    static constexpr index_t AShift=Shift_,LifetimeMode=Mode_;
    static_assert(GroupedInt8NtBypassBLifetimeProblem::MR==16 && GroupedInt8NtBypassBLifetimeProblem::NR==2);
};
}
