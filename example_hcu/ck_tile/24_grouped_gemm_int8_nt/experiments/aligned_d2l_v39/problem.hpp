// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int WM_,int WN_,int Group_=1,bool Native_=false>
struct GroupedInt8NtD2LAlignedProblem : GroupedInt8NtTileProblem<256,256,64,WM_,WN_,
    true,true,true,false,1,true,Group_>
{
    static constexpr bool DoubleBuffer=true,NativeOnly=Native_;
    static constexpr index_t ProducerWaves=0,ComputeThreads=WM_*WN_*64,SwizzleShift=1;
};
}
