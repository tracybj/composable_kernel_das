// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int BM_,int BN_,int BK_,int WM_,int WN_,int Group_=1,int Shift_=1>
struct GroupedInt8NtMediumD2LProblem
{
    using Mmac=WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using CopyVector=ext_vector_t<int32_t,4>;
    static constexpr index_t BM=BM_,BN=BN_,BK=BK_,WM=WM_,WN=WN_;
    static constexpr index_t MGroup=Group_,SwizzleShift=Shift_;
    static constexpr bool Aligned=false,BatchedEpi=false,DoubleBuffer=true;
    static constexpr index_t ProducerWaves=0,ComputeThreads=WM*WN*64,Threads=ComputeThreads;
    static constexpr index_t MR=BM/(WM*16),NR=BN/(WN*16),KVec=BK/16,SwizzleMask=KVec<4?KVec-1:3;
    static_assert(BK==32 || BK==64 || BK==128 || BK==256);
    static_assert(BM%(WM*16)==0 && BN%(WN*16)==0);
};
}
