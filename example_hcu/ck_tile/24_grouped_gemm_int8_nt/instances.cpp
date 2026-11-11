// SPDX-License-Identifier: MIT
#include "kernels/grouped_int8_nt_kernel.hpp"
#include "kernels/grouped_int8_nt_wasp_kernel.hpp"
#include "kernels/grouped_int8_nt_direct.hpp"
#include "kernels/grouped_int8_nt_epilogue.hpp"
#include "instances/grouped_int8_nt_instances.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include <hip/hip_runtime.h>

namespace {

template <class P, class Epilogue = ck_tile::GroupedInt8NtTileEpilogue<P>>
int launch(const void* args, int groups, int m, int n, hipStream_t stream)
{
    dim3 grid(P::MFirst ? (m + P::BM - 1) / P::BM : (n + P::BN - 1) / P::BN,
              P::MFirst ? (n + P::BN - 1) / P::BN : (m + P::BM - 1) / P::BM,
              groups);
    if constexpr(P::MGroup > 0)
    {
        const int mt = (m + P::BM - 1) / P::BM, nt = (n + P::BN - 1) / P::BN;
        grid = dim3(((mt + P::MGroup - 1) / P::MGroup) * P::MGroup * nt, 1, groups);
    }
    ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                           ck_tile::make_kernel<P::Threads, P::MinBlocks>(
                               ck_tile::GroupedInt8NtTileKernel<P, Epilogue>{},
                               grid,
                               dim3(P::Threads),
                               0,
                               static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError() == hipSuccess ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
                                           : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
}

template <class P>
int launch_wasp(const void* args, int groups, int m, int n, hipStream_t stream)
{
    const int mm = P::Transpose ? n : m, nn = P::Transpose ? m : n;
    const int mt = (mm + P::BM - 1) / P::BM, nt = (nn + P::BN - 1) / P::BN;
    dim3 grid(mt * nt, 1, groups);
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<P::Threads, 1>(ck_tile::GroupedInt8NtWaspKernel<P>{},
                                            grid,
                                            dim3(P::Threads),
                                            0,
                                            static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError() == hipSuccess ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
                                           : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
}

} // namespace

extern "C" int ck_tile_grouped_int8_nt_launch(
    unsigned instance_id, const void* args, int g, int m, int n, void* stream)
{
    if(!args || g <= 0 || g > 65535 || m <= 0 || n <= 0)
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
    auto s = reinterpret_cast<hipStream_t>(stream);
    return ck_tile::int8_nt_instances::dispatch_grouped_int8_nt_instance(
        static_cast<ck_tile::GroupedGemmInstanceId>(instance_id),
        [&](auto tag) {
            using P = typename decltype(tag)::type;
            return launch<P>(args, g, m, n, s);
        },
        [&](auto tag) {
            using P = typename decltype(tag)::type;
            return launch_wasp<P>(args, g, m, n, s);
        },
        [&]() {
            dim3 grid((n + 63) / 64, (m + 15) / 16, g);
            ck_tile::launch_kernel(
                ck_tile::stream_config{s, false},
                ck_tile::make_kernel<256, 1>(
                    ck_tile::GroupedInt8NtDirectKernel<>{},
                    grid,
                    dim3(256),
                    0,
                    static_cast<const ck_tile::GemmTransKernelArg*>(args)));
            return hipGetLastError() == hipSuccess
                       ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
                       : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
        },
        [&](auto tag) {
            using P = typename decltype(tag)::type;
            return launch<P, ck_tile::GroupedInt8NtDirectEpilogue<P>>(args, g, m, n, s);
        },
        []() { return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1; });
}
