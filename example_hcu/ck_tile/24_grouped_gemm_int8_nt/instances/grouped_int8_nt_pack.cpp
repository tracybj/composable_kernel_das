// SPDX-License-Identifier: MIT
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
#include "../kernels/grouped_int8_nt_pack_kernel.hpp"
#include "../include/ck_tile_int8_nt_cached.h"

extern "C" int ck_tile_int8_cached_pack_private(
    const ck_tile::Int8CachedWeightPackArgs* args, int g, int n, int k, hipStream_t stream)
{
    using P = ck_tile::Int8CachedWeightPackProblem;
    const dim3 grid((n + 63) / 64, k / 64, g);
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<P::Threads, 1>(
            ck_tile::Int8CachedWeightPackKernel<P>{}, grid, dim3(P::Threads), 0, args));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}
#endif
