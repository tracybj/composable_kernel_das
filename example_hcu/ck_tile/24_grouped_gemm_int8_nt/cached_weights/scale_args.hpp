// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace ck_tile {
struct Int8NtScaleArgs {
    const float* a;
    const float* b;
    int32_t a_stride, b_stride;
};
}
