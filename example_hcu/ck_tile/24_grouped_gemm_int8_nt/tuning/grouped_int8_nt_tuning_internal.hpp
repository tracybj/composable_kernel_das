// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile_int8_nt_tuning.h"
#include <vector>

namespace ck_tile::int8_nt_tuning {

using Candidate = ck_tile_int8_nt_tuning_candidate_v1;

struct Domain
{
    int m, n, k, max_a_stride, max_b_stride;
    bool packed, paired;
};

std::vector<Candidate> enumerate(const Domain& domain, unsigned output_type);

} // namespace ck_tile::int8_nt_tuning
