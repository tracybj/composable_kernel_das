// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace ck_tile {
struct Int8NtKParallelArgs
{
    const int8_t* a;
    const int8_t* b;
    void* c;
    const float* scale_a;
    const float* scale_b;
    int m, n, k;
    int stride_a, stride_b, stride_c;
    int scale_a_stride, scale_b_stride;
};
} // namespace ck_tile
