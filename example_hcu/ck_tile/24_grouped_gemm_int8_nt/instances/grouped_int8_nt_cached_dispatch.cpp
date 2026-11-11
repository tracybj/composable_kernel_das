// SPDX-License-Identifier: MIT
#include "../kernels/grouped_int8_nt_wasp_kernel.hpp"
#include "../kernels/grouped_int8_nt_pack_kernel.hpp"
#include "grouped_int8_nt_instances.hpp"
#include "grouped_int8_nt_routes.hpp"

extern "C" int ck_tile_grouped_int8_nt_launch(
    unsigned instance_id, const void* args, int g, int m, int n, void* stream);

namespace {
template <class P>
int run(const void* args, int g, int m, int n, hipStream_t stream)
{
    const int mm = P::Transpose ? n : m, nn = P::Transpose ? m : n;
    const int mt = (mm + P::BM - 1) / P::BM, nt = (nn + P::BN - 1) / P::BN;
    const dim3 grid(mt * nt, 1, g);
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<P::Threads, 1>(ck_tile::GroupedInt8NtWaspCoreKernel<P>{},
                                            grid,
                                            dim3(P::Threads),
                                            0,
                                            static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}
} // namespace

extern "C" int ck_tile_int8_cached_launch_private(
    unsigned route, unsigned legacy, const void* args, int g, int m, int n, hipStream_t stream)
{
    using namespace ck_tile;
    using namespace ck_tile::int8_nt_instances;
    switch(route)
    {
    case int8_routes::LegacyRaw:
        return ck_tile_grouped_int8_nt_launch(legacy, args, g, m, n, stream);
    case int8_routes::LinearTransposedRaw:
        return run<RawTransposeM256N256>(args, g, m, n, stream);
    case int8_routes::LinearTransposedRecord1024:
        return run<RawTransposeM256N256Record1024>(args, g, m, n, stream);
    case int8_routes::PackedWeightM256N256:
        return run<PackedM256N256>(args, g, m, n, stream);
    case int8_routes::PackedM128N256:
        return run<PackedM128N256>(args, g, m, n, stream);
    case int8_routes::PackedM128N128:
        return run<PackedM128N128>(args, g, m, n, stream);
    case int8_routes::PackedM64N256:
        return run<PackedM64N256>(args, g, m, n, stream);
    case int8_routes::PackedM64N64:
        return run<PackedM64N64>(args, g, m, n, stream);
    case int8_routes::PackedM32N256:
        return run<PackedM32N256>(args, g, m, n, stream);
    case int8_routes::PackedM32N64:
        return run<PackedM32N64>(args, g, m, n, stream);
    case int8_routes::PackedM256N384:
        return run<PackedM256N384>(args, g, m, n, stream);
    default:
        return CK_TILE_INT8_CACHED_UNSUPPORTED_V1;
    }
}

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
