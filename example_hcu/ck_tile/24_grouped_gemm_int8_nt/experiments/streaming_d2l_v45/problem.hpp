// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
template<int Stages_,int Producers_,int Shift_=0,bool Priority_=false,int WM_=4,int WN_=2,int Group_=4,bool Middle_=false>
struct GroupedInt8NtStreamProblem
{
    using Mmac=WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using CopyVector=ext_vector_t<int32_t,4>;
    static constexpr index_t BM=256,BN=256,BK=32,WM=WM_,WN=WN_,MR=BM/(WM*16),NR=BN/(WN*16);
    static constexpr index_t ComputeWaves=WM*WN,ComputeThreads=ComputeWaves*64,ProducerWaves=Producers_;
    static constexpr index_t Threads=ComputeThreads+ProducerWaves*64,Stages=Stages_,SwizzleShift=Shift_,MGroup=Group_;
    static constexpr bool Aligned=true,Priority=Priority_,Middle=Middle_;
    static_assert(Stages>=2 && Stages<=4 && ComputeWaves==8);
    static_assert(Producers_==0 || Producers_==4);
    CK_TILE_DEVICE static bool IsProducer(index_t tid)
    {
        if constexpr(ProducerWaves==0)return false;
        const index_t w=tid/64;return Middle?(w>=4 && w<8):w>=8;
    }
    CK_TILE_DEVICE static index_t StageThread(index_t tid)
    {
        if constexpr(ProducerWaves==0)return tid;
        return (tid/64-(Middle?4:8))*64+tid%64;
    }
    CK_TILE_DEVICE static index_t ComputeWave(index_t tid)
    {
        if(IsProducer(tid))return -1;
        if constexpr(Middle && ProducerWaves>0)return tid/64>=8?tid/64-4:tid/64;
        return tid/64;
    }
};
}
