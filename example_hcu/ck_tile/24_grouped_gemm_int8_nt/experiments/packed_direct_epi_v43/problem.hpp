// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int M,int N,int K,int WM,int WN,bool Vector>
struct GroupedInt8NtPackedDirectEpiProblem : GroupedInt8NtTileProblem<M,N,K,WM,WN,
    false,false,true,true,1,true>
{
    static constexpr bool VectorOutput=Vector;
};
}
