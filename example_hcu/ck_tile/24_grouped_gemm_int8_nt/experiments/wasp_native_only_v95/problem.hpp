// SPDX-License-Identifier: MIT
#pragma once
#include "../wasp_linear_b_v92/epilogue.hpp"
namespace ck_tile {
template<bool Transpose_=false,int RecordCap_=0,int BAddress_=0>
struct GroupedInt8NtWaspNativeOnlyProblem : GroupedInt8NtWaspLinearProblem<Transpose_,8,4,4,true,true,RecordCap_,false,BAddress_> {};
}
