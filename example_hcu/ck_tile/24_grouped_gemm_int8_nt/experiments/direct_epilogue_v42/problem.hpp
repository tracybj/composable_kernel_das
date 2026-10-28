// SPDX-License-Identifier: MIT
#pragma once
#include "../aligned_d2l_v39/problem.hpp"
namespace ck_tile {
template<int WM_,int WN_,int Group_,bool Vector_>
struct GroupedInt8NtDirectEpiProblem : GroupedInt8NtD2LAlignedProblem<WM_,WN_,Group_,true>
{
    static constexpr bool VectorOutput=Vector_;
};
}
