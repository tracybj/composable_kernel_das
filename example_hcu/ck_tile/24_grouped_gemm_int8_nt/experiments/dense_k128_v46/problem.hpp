// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int Producers_=0,bool Swizzle8_=false,bool Paired_=true,bool Aligned_=true,int WM_=4,int WN_=2,int Group_=4,bool Fence_=true>
struct GroupedInt8NtDenseKProblem : GroupedInt8NtTileProblem<256,256,128,WM_,WN_,true,Aligned_,true,false,1,true,Group_>
{
    static constexpr index_t ProducerWaves=Producers_,ComputeThreads=WM_*WN_*64,Threads=ComputeThreads+ProducerWaves*64;
    static constexpr bool Swizzle8=Swizzle8_,Paired=Paired_,Fence=Fence_;
    static_assert(ComputeThreads==512 && (Producers_==0 || Producers_==4));
    static_assert(!(Swizzle8 && Paired),"Paired128-bit reads require the16-byte swizzle");
};
}
