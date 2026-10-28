// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int WM_,int WN_,int MGroup_=1,bool Merged_=false,int Shift_=1>
struct GroupedInt8NtD2LGeometryProblem : GroupedInt8NtTileProblem<256,256,64,WM_,WN_,
    true,false,true,Merged_,1,true,MGroup_>
{
    static constexpr bool DoubleBuffer=true;
    static constexpr index_t ProducerWaves=0,ComputeThreads=WM_*WN_*64;
    static constexpr index_t SwizzleShift=Shift_;
};
}
