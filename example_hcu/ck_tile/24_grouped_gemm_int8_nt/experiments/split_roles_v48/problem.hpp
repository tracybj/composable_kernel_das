// SPDX-License-Identifier: MIT
#pragma once
#include "../dense_k128_v46/problem.hpp"
namespace ck_tile {
template<bool Swizzle8_=false,bool Paired_=true,bool Aligned_=true,int WM_=4,int WN_=2,int Group_=4,bool Priority_=false>
struct GroupedInt8NtSplitRolesProblem : GroupedInt8NtDenseKProblem<4,Swizzle8_,Paired_,Aligned_,WM_,WN_,Group_,true>
{
    static constexpr bool ConsumerPriority=Priority_;
};
}
