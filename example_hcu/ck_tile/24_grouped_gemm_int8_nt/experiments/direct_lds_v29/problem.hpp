// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<bool Double_,bool Merged_,int Producers_=0,int MGroup_=1,int Shift_=0>
struct GroupedInt8NtD2LBankProblem : GroupedInt8NtTileProblem<256,256,64,4,2,
    true,false,true,Merged_,1,true,MGroup_>
{
    static constexpr bool DoubleBuffer=Double_;
    static constexpr int SwizzleShift=Shift_;
    static constexpr index_t ProducerWaves=Producers_;
    static constexpr index_t ComputeThreads=4*2*64;
    static constexpr index_t Threads=ComputeThreads+ProducerWaves*64;
    static_assert(Producers_==0 || Producers_==4);
    static_assert(Producers_==0 || Double_,"Specialized producer uses ping-pong buffers");
};
}
