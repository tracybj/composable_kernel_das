// SPDX-License-Identifier: MIT
#pragma once
#include "../../grouped_int8_nt_wasp_problem.hpp"
namespace ck_tile {
template<bool Transpose,int Batch,int RecordCap,bool Scalar>
struct GroupedInt8NtWaspOutputProblem
    : GroupedInt8NtWaspProblem<Transpose,8,4,Batch,true,true,RecordCap>
{
    static constexpr bool ScalarOutput = Scalar;
};
}
