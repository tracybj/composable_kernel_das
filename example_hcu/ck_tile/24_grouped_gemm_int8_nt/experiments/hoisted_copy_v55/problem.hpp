// SPDX-License-Identifier: MIT
#pragma once
#include "../transposed_bypass_v54/problem.hpp"
namespace ck_tile {
template<bool Aligned_=true,int Group_=4,bool Priority_=false,bool AdvanceResource_=true>
struct GroupedInt8NtHoistedCopyProblem : GroupedInt8NtTransposedProblem<Aligned_,true,Group_,Priority_,2>
{
    static constexpr bool AdvanceResource=AdvanceResource_;
};
}
