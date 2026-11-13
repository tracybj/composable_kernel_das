// SPDX-License-Identifier: MIT
#include "../kernels/grouped_int8_nt_kernel.hpp"
#include "../kernels/grouped_int8_nt_wasp_kernel.hpp"
#include "../kernels/grouped_int8_nt_k_parallel.hpp"
#include "../kernels/grouped_int8_nt_scaled_epilogue.hpp"
#include "grouped_int8_nt_instances.hpp"
#include "grouped_int8_nt_routes.hpp"
#ifdef CK_INT8_CACHED_TUNING
#include "../tuning/grouped_int8_nt_tuning_internal.hpp"
#endif

namespace {

template <bool BF>
using PairedM256N384Problem = std::conditional_t<
    BF,
    ck_tile::GroupedInt8NtWaspPublicWideNProblem<false, 12, 1, 2, true, true, 0, false, true, 0, 12>,
    ck_tile::int8_nt_instances::PackedM256N384>;

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
#define CK_INT8_SCALED_COOPERATIVE(id, ...) \
    case int8_routes::id: return cooperative<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_SCALED_WASP(id, ...) \
    case int8_routes::id: return wasp<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_SCALED_SCALED_WASP(id, ...) CK_INT8_SCALED_WASP(id, __VA_ARGS__)
#define CK_INT8_SCALED_PAIRED_WASP(id, ...) \
    case int8_routes::id: return paired_wasp<BF, __VA_ARGS__>(args, scales, g, m, n, stream);
#define CK_INT8_SCALED_K_PARALLEL(id, ...) \
    case int8_routes::id: \
        return m == 1 ? k_parallel<BF, 1>(args, g, n, stream) \
                      : k_parallel<BF, 2>(args, g, n, stream);
#define CK_INT8_ROUTE(id, abi, kind, weights, ...) CK_INT8_SCALED_##kind(id, __VA_ARGS__)
#include "grouped_int8_nt_routes.inc"
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
template <class P, bool Packed = false>
void add_tile(std::vector<Candidate>& out, const Domain& d, unsigned route, unsigned instance)
{
    if constexpr(Packed)
    {
        if(!d.packed || d.n % P::BN != 0 || d.k % P::BK != 0)
            return;
    }
    if constexpr(P::Aligned)
        if(d.m % P::BM != 0 || d.n % P::BN != 0 || d.k % P::BK != 0)
            return;
    out.push_back({route, instance, P::BM, P::BN, P::BK, P::Threads, Packed});
}

template <class P, bool Paired = false>
void add_wasp(std::vector<Candidate>& out, const Domain& d, unsigned route, unsigned instance)
{
    if((P::PackedB && !d.packed) || (Paired && !d.paired))
        return;
    const int m = P::Transpose ? d.n : d.m;
    const int n = P::Transpose ? d.m : d.n;
    const int a_stride = P::Transpose ? d.max_b_stride : d.max_a_stride;
    const int b_stride = P::Transpose ? d.max_a_stride : (P::PackedB ? d.k : d.max_b_stride);
    // Match the native geometry in GroupedInt8NtWaspKernel::is_native.
    constexpr int ma = P::Transpose && !P::PackedB && P::Pipeline::MasksInactiveN ? 16 : P::BM;
    constexpr int na = P::BN == 384 || P::Pipeline::MasksInactiveN ? 32 : P::BN;
    if(m % ma != 0 || n % na != 0 || d.k % P::BK != 0 || b_stride >= 16384 ||
       int64_t(P::BM < m ? P::BM : m) * a_stride > INT32_MAX ||
       int64_t(n) * b_stride > INT32_MAX)
        return;
    out.push_back({route, instance, P::BM, P::BN, P::BK, P::Threads, P::PackedB});
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
#include "grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
#define CK_INT8_TUNING_COOPERATIVE(id, ...) \
    add_tile<__VA_ARGS__, int8_routes::uses_packed_weights(int8_routes::id)>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_WASP(id, ...) add_wasp<__VA_ARGS__>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_SCALED_WASP(id, ...) CK_INT8_TUNING_WASP(id, __VA_ARGS__)
#define CK_INT8_TUNING_PAIRED_WASP(id, ...) add_wasp<__VA_ARGS__, true>(out, d, int8_routes::id, 0);
#define CK_INT8_TUNING_K_PARALLEL(id, ...) \
    if(d.m <= 2) \
        out.push_back({int8_routes::id, 0, d.m, __VA_ARGS__::Cols, __VA_ARGS__::KSpan, __VA_ARGS__::Threads, 0});
#define CK_INT8_ROUTE(id, abi, kind, weights, ...) CK_INT8_TUNING_##kind(id, __VA_ARGS__)
#include "grouped_int8_nt_routes.inc"
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
