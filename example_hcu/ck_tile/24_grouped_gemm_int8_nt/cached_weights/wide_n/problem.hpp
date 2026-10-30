// SPDX-License-Identifier: MIT
#pragma once
#include "../core_problem.hpp"
namespace ck_tile {
template<bool Transpose_=false,int Split_=8,int Group_=4,int Batch_=4,bool Prefetch_=true,bool OneBarrier_=true,int RecordCap_=0,bool EarlyB_=false,bool PackedB_=false,int BAddress_=0,int ConsumerWaves_=8>
using GroupedInt8NtWaspPublicWideNProblem = GroupedInt8NtWaspCoreProblem<256,Transpose_,Split_,Group_,Batch_,Prefetch_,OneBarrier_,RecordCap_,EarlyB_,PackedB_,BAddress_,ConsumerWaves_>;
}
