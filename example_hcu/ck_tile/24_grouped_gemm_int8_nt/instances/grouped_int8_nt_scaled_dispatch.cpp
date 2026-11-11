// SPDX-License-Identifier: MIT
#include "../kernels/grouped_int8_nt_kernel.hpp"
#include "../kernels/grouped_int8_nt_wasp_kernel.hpp"
#include "../kernels/grouped_int8_nt_k_parallel.hpp"
#include "../kernels/grouped_int8_nt_scaled_epilogue.hpp"
#include "grouped_int8_nt_instances.hpp"
#include "grouped_int8_nt_routes.hpp"

namespace {

template <bool BF, class P, class Epi = ck_tile::GroupedInt8NtScaledWaspEpilogue<P, BF>>
int wasp(const void* args, const void* scales, int g, int m, int n, hipStream_t stream)
{
    const int mm = P::Transpose ? n : m, nn = P::Transpose ? m : n;
    const int mt = (mm + P::BM - 1) / P::BM, nt = (nn + P::BN - 1) / P::BN;
    const dim3 grid(((mt + P::MGroup - 1) / P::MGroup) * P::MGroup * nt, 1, g);
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<P::Threads, 1>(ck_tile::GroupedInt8NtScaledWaspKernel<P, BF, Epi>{},
                                            grid,
                                            dim3(P::Threads),
                                            0,
                                            static_cast<const ck_tile::GemmTransKernelArg*>(args),
                                            static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}

template <bool BF, class P>
int cooperative(const void* args, const void* scales, int groups, int m, int n, hipStream_t stream)
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
                               ck_tile::GroupedInt8NtScaledTileKernel<P, BF>{},
                               grid,
                               dim3(P::Threads),
                               0,
                               static_cast<const ck_tile::GemmTransKernelArg*>(args),
                               static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}

template <bool BFloat>
int raw(unsigned legacy, const void* args, const void* scales, int g, int m, int n, hipStream_t s)
{
    using namespace ck_tile::int8_nt_instances;
    auto launch_tile = [&](auto tag) {
        using P = typename decltype(tag)::type;
        return cooperative<BFloat, P>(args, scales, g, m, n, s);
    };
    auto launch_default = [&]() {
        return cooperative<BFloat, DefaultFallbackTileProblem>(args, scales, g, m, n, s);
    };
    return dispatch_grouped_int8_nt_instance(
        static_cast<ck_tile::GroupedGemmInstanceId>(legacy),
        launch_tile,
        [&](auto tag) {
            using P = typename decltype(tag)::type;
            return wasp<BFloat, P>(args, scales, g, m, n, s);
        },
        launch_default,
        launch_tile,
        launch_default);
}

template <bool BF, int Rows>
int k_parallel(const void* args, int g, int n, hipStream_t stream)
{
    using K = ck_tile::GroupedInt8NtScaledKParallelKernel<Rows, BF>;
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream, false},
        ck_tile::make_kernel<512, 1>(K{},
                                     dim3((n + 3) / 4, 1, g),
                                     dim3(512),
                                     0,
                                     static_cast<const ck_tile::Int8NtKParallelArgs*>(args)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}

template <bool BF, class P>
int paired_wasp(const void* args, const void* scales, int g, int m, int n, hipStream_t stream)
{
    return wasp<BF, P, ck_tile::GroupedInt8NtPairedStoreEpilogue<P, BF>>(
        args, scales, g, m, n, stream);
}

template <bool BF>
int dispatch(unsigned route,
             unsigned legacy,
             const void* args,
             const void* scales,
             int g,
             int m,
             int n,
             hipStream_t stream)
{
    using namespace ck_tile;
    using namespace ck_tile::int8_nt_instances;
    switch(route)
    {
    case int8_routes::LegacyRaw: return raw<BF>(legacy, args, scales, g, m, n, stream);
    case int8_routes::LinearTransposedRaw:
        return wasp<BF, RawTransposeM256N256>(args, scales, g, m, n, stream);
    case int8_routes::LinearTransposedRecord1024:
        return wasp<BF, RawTransposeM256N256Record1024>(args, scales, g, m, n, stream);
    case int8_routes::PackedWeightM256N256:
        return wasp<BF, PackedM256N256>(args, scales, g, m, n, stream);
    case int8_routes::PackedM128N256:
        return wasp<BF, PackedM128N256>(args, scales, g, m, n, stream);
    case int8_routes::PackedM128N128:
        return wasp<BF, PackedM128N128>(args, scales, g, m, n, stream);
    case int8_routes::PackedM64N256:
        return wasp<BF, PackedM64N256>(args, scales, g, m, n, stream);
    case int8_routes::PackedM64N64:
        return wasp<BF, PackedM64N64>(args, scales, g, m, n, stream);
    case int8_routes::PackedM32N256:
        return wasp<BF, PackedM32N256>(args, scales, g, m, n, stream);
    case int8_routes::PackedM32N64:
        return wasp<BF, PackedM32N64>(args, scales, g, m, n, stream);
    case int8_routes::PackedM256N384:
        return wasp<BF, PackedM256N384>(args, scales, g, m, n, stream);
    case int8_routes::KParallelDecode:
        return m == 1 ? k_parallel<BF, 1>(args, g, n, stream)
                      : k_parallel<BF, 2>(args, g, n, stream);
    case int8_routes::PackedCooperativeM1N256:
        return cooperative<BF, PackedM16N64K256>(args, scales, g, m, n, stream);
    case int8_routes::PackedM256N384MaskedM4:
        return wasp<BF, ScaledPackedM256N384MaskedM4>(args, scales, g, m, n, stream);
    case int8_routes::PackedM256N384MaskedM8:
        return wasp<BF, ScaledPackedM256N384MaskedM8>(args, scales, g, m, n, stream);
    case int8_routes::PairedM256N256:
        return paired_wasp<BF, PackedM256N256>(args, scales, g, m, n, stream);
    case int8_routes::PairedM256N384:
        return paired_wasp<BF, PackedM256N384>(args, scales, g, m, n, stream);
    case int8_routes::PairedM256N384MaskedM4:
        return paired_wasp<BF, ScaledPackedM256N384MaskedM4>(args, scales, g, m, n, stream);
    case int8_routes::PairedM256N384MaskedM8:
        return paired_wasp<BF, ScaledPackedM256N384MaskedM8>(args, scales, g, m, n, stream);
    case int8_routes::PairedM128N256:
        return paired_wasp<BF, PackedM128N256>(args, scales, g, m, n, stream);
    default: return CK_TILE_INT8_CACHED_UNSUPPORTED_V1;
    }
}

} // namespace

extern "C" int ck_tile_int8_scaled_launch_private(unsigned output,
                                                  unsigned route,
                                                  unsigned legacy,
                                                  const void* args,
                                                  const void* scales,
                                                  int g,
                                                  int m,
                                                  int n,
                                                  hipStream_t stream)
{
    return output == ck_tile::int8_output_types::Bf16
               ? dispatch<true>(route, legacy, args, scales, g, m, n, stream)
               : dispatch<false>(route, legacy, args, scales, g, m, n, stream);
}
