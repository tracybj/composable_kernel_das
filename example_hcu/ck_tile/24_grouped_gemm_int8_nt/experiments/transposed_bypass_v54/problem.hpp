// SPDX-License-Identifier: MIT
#pragma once
#include "../bypass_b_ordered_v52/problem.hpp"
namespace ck_tile {
template<bool Aligned_=true,bool PrefetchB_=true,int Group_=4,bool Priority_=false,int Mode_=2>
struct GroupedInt8NtTransposedProblem : GroupedInt8NtBypassBOrderedProblem<Aligned_,PrefetchB_,Group_,Priority_,0,Mode_>
{};
}
