// SPDX-License-Identifier: MIT
#pragma once
#include "../../../grouped_int8_nt_problem.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "pipeline.hpp"
namespace ck_tile {
template<bool Transpose_=false,int Split_=6,int Group_=1,int Batch_=4,bool Prefetch_=true,bool OneBarrier_=true,int RecordCap_=0,bool EarlyB_=false,bool PackedB_=false,int BAddress_=0,int ConsumerWaves_=6>
struct GroupedInt8NtWaspOccupancyM128Problem : GroupedInt8NtTileProblem<128,32*ConsumerWaves_,128,1,ConsumerWaves_,true,true,true,false,1,true,Group_>
{
    static constexpr bool Transpose=Transpose_,PackedB=PackedB_;
    static constexpr index_t ProducerWaves=2,ComputeThreads=64*ConsumerWaves_,Threads=64*(2+ConsumerWaves_);
    using Shape=TileGemmShape<sequence<128,32*ConsumerWaves_,128>,sequence<1,ConsumerWaves_,1>,sequence<16,16,32>>;
    using Row=tensor_layout::gemm::RowMajor;using Col=tensor_layout::gemm::ColumnMajor;
    using Traits=PersistentTileGemmUniversalTraits<true,false,false,false,Row,Col,Row,
        false,false,false,true,false,false>;
    using PipelineProblem=UniversalGemmPipelineProblem<int8_t,int8_t,int32_t,Shape,Traits,
        GemmPipelineScheduler::Intrawave,true,TailNumber::Full,null_type,null_type,false,2,2>;
    using Pipeline=GemmPipelineInt8WaspOccupancyM128<PipelineProblem,GemmPipelineAgBgCrCompV4DefaultPolicy,Split_,Batch_,Prefetch_,OneBarrier_,RecordCap_,EarlyB_,PackedB_,BAddress_>;
    static_assert(Pipeline::BlockSize==Threads && Pipeline::kWaspLdsBytes==32768);
};
}
