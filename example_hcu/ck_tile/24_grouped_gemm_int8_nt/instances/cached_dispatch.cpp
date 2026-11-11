// SPDX-License-Identifier: MIT
#include "../kernels/core_kernel.hpp"
#include "instances.hpp"
#include "routes.hpp"
#include "../kernels/pack_kernel.hpp"

extern "C" int ck_tile_grouped_int8_nt_launch(
    unsigned instance_id, const void* args, int g, int m, int n, void* stream);
#ifdef CK_INT8_CACHED_TUNING
extern "C" int ck_tile_int8_operand_launch_private(
    unsigned, unsigned, const void*, const void*, int, int, int, hipStream_t);
extern "C" int ck_tile_int8_packed_small_launch_private(
    unsigned, unsigned, const void*, const void*, int, int, int, hipStream_t);
extern "C" int ck_tile_int8_experimental_wasp_launch_private(
    unsigned, unsigned, const void*, const void*, int, int, int, hipStream_t);
#endif
namespace {
template <class P>
int run(const void* args, int g, int m, int n, hipStream_t stream)
{
    const int mm = P::Transpose ? n : m, nn = P::Transpose ? m : n;
    const int mt = (mm + P::BM - 1) / P::BM, nt = (nn + P::BN - 1) / P::BN;
    const dim3 grid(((mt + P::MGroup - 1) / P::MGroup) * P::MGroup * nt, 1, g);
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
#ifdef CK_INT8_CACHED_TUNING
    if(route == 0 && legacy >= 9001 && legacy <= 9071)
        return ck_tile_int8_operand_launch_private(0, legacy, args, nullptr, g, m, n, stream);
    if((route >= 32 && route <= 43) || (route >= 80 && route <= 83))
        return ck_tile_int8_packed_small_launch_private(0, route, args, nullptr, g, m, n, stream);
    if((route >= 13 && route <= 21) || (route >= 48 && route <= 79))
        return ck_tile_int8_experimental_wasp_launch_private(
            0, route, args, nullptr, g, m, n, stream);
#endif
    switch(route)
    {
    case int8_routes::LegacyRaw:
        return ck_tile_grouped_int8_nt_launch(legacy, args, g, m, n, stream);
#ifdef CK_INT8_CACHED_TUNING
    case int8_routes::LinearRaw:
        return run<ck_tile::int8_nt_instances::RawM256N256>(args, g, m, n, stream);
    case int8_routes::LinearRawRecord1024:
        return run<ck_tile::int8_nt_instances::RawM256N256Record1024>(args, g, m, n, stream);
#endif
    case int8_routes::LinearTransposedRaw:
        return run<ck_tile::int8_nt_instances::RawTransposeM256N256>(args, g, m, n, stream);
    case int8_routes::LinearTransposedRecord1024:
        return run<ck_tile::int8_nt_instances::RawTransposeM256N256Record1024>(
            args, g, m, n, stream);
    case int8_routes::PackedWeightM256N256:
        return run<ck_tile::int8_nt_instances::PackedM256N256>(args, g, m, n, stream);
    case int8_routes::PackedM128N256:
        return run<ck_tile::int8_nt_instances::PackedM128N256>(args, g, m, n, stream);
    case int8_routes::PackedM128N128:
        return run<ck_tile::int8_nt_instances::PackedM128N128>(args, g, m, n, stream);
    case int8_routes::PackedM64N256:
        return run<ck_tile::int8_nt_instances::PackedM64N256>(args, g, m, n, stream);
    case int8_routes::PackedM64N64:
        return run<ck_tile::int8_nt_instances::PackedM64N64>(args, g, m, n, stream);
    case int8_routes::PackedM32N256:
        return run<ck_tile::int8_nt_instances::PackedM32N256>(args, g, m, n, stream);
    case int8_routes::PackedM32N64:
        return run<ck_tile::int8_nt_instances::PackedM32N64>(args, g, m, n, stream);
    case int8_routes::PackedM256N384:
        return run<ck_tile::int8_nt_instances::PackedM256N384>(args, g, m, n, stream);
#ifdef CK_INT8_CACHED_TUNING
    case int8_routes::PackedM256N384MaskedM4:
        return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM4>(args, g, m, n, stream);
    case int8_routes::PackedM256N384MaskedM8:
        return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM8>(args, g, m, n, stream);
#endif
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
