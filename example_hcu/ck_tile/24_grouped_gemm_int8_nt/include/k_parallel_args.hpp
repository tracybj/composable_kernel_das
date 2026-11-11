// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace ck_tile {
struct Int8NtKParallelArgs
{
    const int8_t* a;
    const int8_t* b;
    void* c;
    const float* sa;
    const float* sb;
    int m, n, k, as, bs, cs, sas, sbs;
};
} // namespace ck_tile
