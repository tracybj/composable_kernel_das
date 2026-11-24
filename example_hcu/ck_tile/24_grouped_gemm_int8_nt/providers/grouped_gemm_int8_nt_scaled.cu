// SPDX-License-Identifier: MIT
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__)
#include "../include/grouped_gemm_int8_nt_device_args.h"
#include "../kernels/grouped_int8_nt_kernel.hpp"
#include "../kernels/grouped_int8_nt_wasp_kernel.hpp"
#include "../kernels/grouped_int8_nt_k_parallel.hpp"
#include "../kernels/grouped_int8_nt_scaled_epilogue.hpp"
#include "../instances/grouped_int8_nt_instances.hpp"
#include "../instances/grouped_int8_nt_routes.hpp"
#ifdef CK_INT8_CACHED_TUNING
#include "../tuning/grouped_int8_nt_tuning_internal.hpp"
#endif

#include <cstddef>

static_assert(sizeof(ck_tile_hcu_grouped_gemm_int8_scale_v2) == sizeof(ck_tile::Int8NtScaleArgs),
              "C API scale descriptor size must match Int8NtScaleArgs");
static_assert(offsetof(ck_tile_hcu_grouped_gemm_int8_scale_v2, a) ==
                      offsetof(ck_tile::Int8NtScaleArgs, a) &&
                  offsetof(ck_tile_hcu_grouped_gemm_int8_scale_v2, b) ==
                      offsetof(ck_tile::Int8NtScaleArgs, b) &&
                  offsetof(ck_tile_hcu_grouped_gemm_int8_scale_v2, a_stride) ==
                      offsetof(ck_tile::Int8NtScaleArgs, a_stride) &&
                  offsetof(ck_tile_hcu_grouped_gemm_int8_scale_v2, b_stride) ==
                      offsetof(ck_tile::Int8NtScaleArgs, b_stride),
              "C API scale descriptor layout must match Int8NtScaleArgs");

namespace {

// The BF16 and FP16 BN384 instances use M clusters of 1 and 4 respectively.
template <bool BF>
using RawPairedM256N384Problem = ck_tile::int8_nt_instances::RawPairedM256N384Problem<BF>;

// =============================================================================
// Three Standard Launch Providers (1:1 mapped to Device Architectures)
//   1. WASP Architecture:        wasp(...)
//      - Asynchronous Direct-to-LDS Producer-Consumer pipeline
//      - Parameterized Epilogue: default single-store vs paired-store
//   2. Cooperative Architecture: cooperative(...)
//      - Synchronous Wave-Cooperative tile pipeline
//      - Parameterized Pipeline: standard LDS-staging vs 0-LDS direct-VGPR
//   3. K-Parallel Architecture:  k_parallel(...)
//      - Multi-Wave split-K reduction for extreme small-M (M=1, 2) decode
// =============================================================================

template <bool BF, class P, class Epi = ck_tile::GroupedInt8NtScaledWaspEpilogue<P, BF>>
int wasp(const void* args, const void* scales, int g, int m, int n, hipStream_t stream)
{
    const int mm = P::Transpose ? n : m, nn = P::Transpose ? m : n;
    const int mt = (mm + P::BM - 1) / P::BM, nt = (nn + P::BN - 1) / P::BN;
    const dim3 grid(mt * nt, 1, g);
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
    using Kernel = ck_tile::GroupedInt8NtScaledTileKernel<P, BF>;
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
                               Kernel{},
                               grid,
                               dim3(P::Threads),
                               0,
                               static_cast<const ck_tile::GemmTransKernelArg*>(args),
                               static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}

template <bool BF, int Rows>
int k_parallel(const void* args, const void* scales, int g, int n, hipStream_t stream)
{
    using K = ck_tile::GroupedInt8NtScaledKParallelKernel<Rows, BF>;
    const ck_tile::stream_config config{stream, false};
    const dim3 grid((n + 3) / 4, 1, g);
    const dim3 threads(512);
    ck_tile::launch_kernel(
        config,
        ck_tile::make_kernel<512, 1>(K{},
                                     grid,
                                     threads,
                                     0,
                                     static_cast<const ck_tile::GemmTransKernelArg*>(args),
                                     static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
    return hipGetLastError() == hipSuccess ? CK_TILE_INT8_CACHED_SUCCESS_V1
                                           : CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
}

template <bool BF>
int dispatch(unsigned instance,
             const void* args,
             const void* scales,
             int g,
             int m,
             int n,
             hipStream_t stream)
{
    using namespace ck_tile;
    using namespace ck_tile::int8_nt_instances;
    switch(instance)
    {
#define CK_INT8_LEGACY_TILE(id, ...)                       \
    case static_cast<unsigned>(GroupedGemmInstanceId::id): \
        return cooperative<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_LEGACY_WASP(id, ...)                       \
    case static_cast<unsigned>(GroupedGemmInstanceId::id): \
        return wasp<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_LEGACY_DIRECT(id) CK_INT8_LEGACY_TILE(id, DefaultFallbackTileProblem)
#define CK_INT8_LEGACY_DIRECT_EPI(id, ...) CK_INT8_LEGACY_TILE(id, __VA_ARGS__)
#include "../instances/grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
#define CK_INT8_SCALED_COOPERATIVE(id, ...)              \
    case int8_instance_ids::from_route(int8_routes::id): \
        return cooperative<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_SCALED_WASP(id, ...)                     \
    case int8_instance_ids::from_route(int8_routes::id): \
        return wasp<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_SCALED_SCALED_WASP(id, ...) CK_INT8_SCALED_WASP(id, __VA_ARGS__)
#define CK_INT8_SCALED_PAIRED_WASP(id, ...)                                                     \
    case int8_instance_ids::from_route(int8_routes::id):                                        \
        return wasp<BF, __VA_ARGS__, ck_tile::GroupedInt8NtPairedStoreEpilogue<__VA_ARGS__, BF>>( \
            args, scales, g, m, n, stream);
#define CK_INT8_SCALED_K_PARALLEL(id, ...)                    \
    case int8_instance_ids::from_route(int8_routes::id, 1):   \
        return k_parallel<BF, 1>(args, scales, g, n, stream); \
    case int8_instance_ids::from_route(int8_routes::id, 2):   \
        return k_parallel<BF, 2>(args, scales, g, n, stream); \
    case int8_instance_ids::from_route(int8_routes::id, 4):   \
        return k_parallel<BF, 4>(args, scales, g, n, stream); \
    case int8_instance_ids::from_route(int8_routes::id, 8):   \
        return k_parallel<BF, 8>(args, scales, g, n, stream);
#define CK_INT8_ROUTE(id, abi, kind, ...) CK_INT8_SCALED_##kind(id, __VA_ARGS__)
#include "../instances/grouped_int8_nt_routes.inc"
#undef CK_INT8_ROUTE
#undef CK_INT8_SCALED_K_PARALLEL
#undef CK_INT8_SCALED_PAIRED_WASP
#undef CK_INT8_SCALED_SCALED_WASP
#undef CK_INT8_SCALED_WASP
#undef CK_INT8_SCALED_COOPERATIVE
    default: return CK_TILE_INT8_CACHED_UNSUPPORTED_V1;
    }
}

} // namespace

#ifdef CK_INT8_CACHED_TUNING
namespace ck_tile::int8_nt_tuning {
namespace {
template <class P>
void add_tile(std::vector<Candidate>& out, const Domain& d, unsigned route, unsigned instance)
{
    if constexpr(P::Aligned)
        if(d.m % P::BM != 0 || d.n % P::BN != 0 || d.k % P::BK != 0)
            return;
    out.push_back({route, instance, P::BM, P::BN, P::BK, P::Threads, 0});
}

template <class P, bool Paired = false>
void add_wasp(std::vector<Candidate>& out, const Domain& d, unsigned route, unsigned instance)
{
    if(Paired && !d.paired)
        return;
    const int m        = P::Transpose ? d.n : d.m;
    const int n        = P::Transpose ? d.m : d.n;
    const int a_stride = P::Transpose ? d.max_b_stride : d.max_a_stride;
    const int b_stride = P::Transpose ? d.max_a_stride : d.max_b_stride;
    // Match the native geometry in GroupedInt8NtWaspKernel::is_native.
    constexpr int ma = P::Transpose && P::Pipeline::MasksInactiveN ? 16 : P::BM;
    constexpr int na = P::BN == 384 || P::Pipeline::MasksInactiveN ? 32 : P::BN;
    if(m % ma != 0 || n % na != 0 || d.k % P::BK != 0 || (P::Pipeline::BAddressMode == 0 && b_stride >= 16384) ||
       int64_t(P::BM < m ? P::BM : m) * a_stride > INT32_MAX || int64_t(n) * b_stride > INT32_MAX)
        return;
    out.push_back({route, instance, P::BM, P::BN, P::BK, P::Threads, 0});
}

template <bool BF>
std::vector<Candidate> enumerate_for_output(const Domain& d)
{
    using namespace ck_tile::int8_nt_instances;
    using Instance = GroupedGemmInstanceId;
    std::vector<Candidate> out;
    // NT loads require full 16-byte K vectors. The tile epilogue stores groups
    // of four columns. These restrictions cover the scaled benchmark contract.
    if(d.m <= 0 || d.n % 4 != 0 || d.k % 16 != 0)
        return out;
#define CK_INT8_LEGACY_TILE(id, ...) \
    add_tile<__VA_ARGS__>(out, d, int8_routes::LegacyRaw, static_cast<unsigned>(Instance::id));
#define CK_INT8_LEGACY_WASP(id, ...) \
    add_wasp<__VA_ARGS__>(out, d, int8_routes::LegacyRaw, static_cast<unsigned>(Instance::id));
#define CK_INT8_LEGACY_DIRECT(id) CK_INT8_LEGACY_TILE(id, DefaultFallbackTileProblem)
#define CK_INT8_LEGACY_DIRECT_EPI(id, ...) CK_INT8_LEGACY_TILE(id, __VA_ARGS__)
#include "../instances/grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
#define CK_INT8_TUNING_COOPERATIVE(id, ...) add_tile<__VA_ARGS__>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_WASP(id, ...) add_wasp<__VA_ARGS__>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_SCALED_WASP(id, ...) CK_INT8_TUNING_WASP(id, __VA_ARGS__)
#define CK_INT8_TUNING_PAIRED_WASP(id, ...) add_wasp<__VA_ARGS__, true>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_K_PARALLEL(id, ...)   \
    if(d.m <= 2)                             \
        out.push_back({int8_routes::id,      \
                       0,                    \
                       d.m,                  \
                       __VA_ARGS__::Cols,    \
                       __VA_ARGS__::KSpan,   \
                       __VA_ARGS__::Threads, \
                       0});
#define CK_INT8_ROUTE(id, abi, kind, ...) CK_INT8_TUNING_##kind(id, __VA_ARGS__)
#include "../instances/grouped_int8_nt_routes.inc"
#undef CK_INT8_ROUTE
#undef CK_INT8_TUNING_K_PARALLEL
#undef CK_INT8_TUNING_PAIRED_WASP
#undef CK_INT8_TUNING_SCALED_WASP
#undef CK_INT8_TUNING_WASP
#undef CK_INT8_TUNING_COOPERATIVE
    return out;
}
} // namespace

std::vector<Candidate> enumerate(const Domain& domain, unsigned output_type)
{
    return output_type == int8_output_types::Bf16 ? enumerate_for_output<true>(domain)
                                                  : enumerate_for_output<false>(domain);
}
} // namespace ck_tile::int8_nt_tuning
#endif

extern "C" std::size_t ck_tile_int8_nt_device_args_size_private()
{
    return sizeof(ck_tile::GemmTransKernelArg);
}

extern "C" int ck_tile_int8_scaled_launch_private(unsigned output,
                                                  unsigned instance,
                                                  const void* args,
                                                  const void* scales,
                                                  int g,
                                                  int m,
                                                  int n,
                                                  hipStream_t stream)
{
    return output == ck_tile::int8_output_types::Bf16
               ? dispatch<true>(instance, args, scales, g, m, n, stream)
               : dispatch<false>(instance, args, scales, g, m, n, stream);
}
#endif
