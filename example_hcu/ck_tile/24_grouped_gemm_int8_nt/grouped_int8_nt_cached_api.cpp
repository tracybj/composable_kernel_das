// SPDX-License-Identifier: MIT
#include "include/ck_tile_int8_nt_cached.h"
#include "instances/grouped_int8_nt_routes.hpp"
#include "instances/grouped_int8_nt_selector.hpp"
#include "include/grouped_int8_nt_scale_args.hpp"
#ifdef CK_INT8_CACHED_TUNING
#include "tuning/grouped_int8_nt_tuning_internal.hpp"
#endif
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int ck_tile_int8_scaled_launch_private(
    unsigned, unsigned, const void*, const void*, int, int, int, hipStream_t);

namespace {
thread_local std::string last_error;
struct ApiError : std::runtime_error
{
    int status;
    ApiError(int s, const char* message) : std::runtime_error(message), status(s) {}
};
void require(bool ok, int status, const char* message)
{
    if(!ok)
        throw ApiError(status, message);
}
void hip_check(hipError_t error)
{
    if(error != hipSuccess)
        throw ApiError(CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1, hipGetErrorString(error));
}
template <class F>
int api(F f)
{
    try
    {
        last_error.clear();
        f();
        return CK_TILE_INT8_CACHED_SUCCESS_V1;
    }
    catch(const ApiError& e)
    {
        last_error = e.what();
        return e.status;
    }
    catch(const std::exception& e)
    {
        last_error = e.what();
        return CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
    }
    catch(...)
    {
        last_error = "unknown INT8 cached API failure";
        return CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;
    }
}
template <class T>
void header(const T* value)
{
    require(value != nullptr, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null descriptor");
    require(value->abi_version == 1,
            CK_TILE_INT8_CACHED_BAD_VERSION_V1,
            "unsupported cached ABI version");
    require(value->struct_size == sizeof(T),
            CK_TILE_INT8_CACHED_BAD_STRUCT_SIZE_V1,
            "wrong cached descriptor size");
    require(value->reserved[0] == 0 && value->reserved[1] == 0,
            CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
            "nonzero reserved descriptor field");
}
bool aligned(const void* p, size_t alignment)
{
    return (reinterpret_cast<uintptr_t>(p) & (alignment - 1)) == 0;
}
void current_device_is(int expected)
{
    int device = -1;
    hip_check(hipGetDevice(&device));
    require(device == expected,
            CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
            "cached handle belongs to another current device");
}
using KArg = ck_tile::GemmTransKernelArg;
} // namespace

struct ck_tile_int8_nt_cached_weights_impl
{
    int device = -1, g = 0, n = 0, k = 0, compute_units = 0;
    std::vector<ck_tile_int8_nt_cached_weight_desc_v1> raw;
    hipEvent_t ready = nullptr;
    std::atomic<unsigned> plans{0};
    ~ck_tile_int8_nt_cached_weights_impl()
    {
        if(ready)
            (void)hipEventDestroy(ready);
    }
};
struct ck_tile_int8_nt_cached_plan_impl
{
    ck_tile_int8_nt_cached_weights_impl* weights;
    void* args = nullptr;
    int g = 0, max_m = 0, launch_m = 1, n = 0, k = 0;
    unsigned instance_key = 0, legacy_instance = 0;
    const unsigned output_type;
    ck_tile::Int8NtScaleArgs* scales = nullptr;
    std::vector<hipStream_t> streams;
    std::mutex mutex;
    ck_tile_int8_nt_cached_plan_impl(ck_tile_int8_nt_cached_weights_impl* w, unsigned type)
        : weights(w), output_type(type)
    {
        weights->plans.fetch_add(1);
    }
    ~ck_tile_int8_nt_cached_plan_impl()
    {
        for(auto stream : streams)
            (void)hipStreamSynchronize(stream);
        if(args)
            (void)hipFree(args);
        if(scales)
            (void)hipFree(scales);
        weights->plans.fetch_sub(1);
    }
};

extern "C" CK_TILE_INT8_CACHED_API const char* ck_tile_int8_nt_cached_last_error_v1()
{
    return last_error.c_str();
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_weights_create_v1(const ck_tile_int8_nt_cached_weight_desc_v1* source,
                                         int32_t g,
                                         void* opaque_stream,
                                         ck_tile_int8_nt_cached_weights_handle_v1* result)
{
    return api([&] {
        require(result != nullptr, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null weights result");
        *result = nullptr;
        require(source && g > 0 && g <= 65535,
                CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                "invalid weights/group count");
        int device = -1;
        hip_check(hipGetDevice(&device));
        hipDeviceProp_t prop{};
        hip_check(hipGetDeviceProperties(&prop, device));
        require(std::strncmp(prop.gcnArchName, "gfx936", 6) == 0,
                CK_TILE_INT8_CACHED_UNSUPPORTED_ARCH_V1,
                "INT8 cached kernels require gfx936");
        for(int i = 0; i < g; ++i)
        {
            const auto& w = source[i];
            header(&w);
            require(w.reserved0 == 0 && w.data && aligned(w.data, 16) && w.n > 0 &&
                        w.n <= 1048576 && w.n % 4 == 0 && w.k >= 256 && w.k <= 131056 &&
                        w.k % 16 == 0 && w.row_stride >= w.k && w.row_stride % 16 == 0,
                    CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                    "invalid signed INT8 NT weight descriptor/alignment");
            require(w.n == source[0].n && w.k == source[0].k,
                    CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                    "v1 cached weights require common N/K");
            require(int64_t(w.n) * w.row_stride <= std::numeric_limits<int32_t>::max(),
                    CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                    "weight span exceeds INT8 buffer domain");
        }
        auto w           = std::make_unique<ck_tile_int8_nt_cached_weights_impl>();
        w->device        = device;
        w->compute_units = prop.multiProcessorCount;
        w->g             = g;
        w->n             = source[0].n;
        w->k             = source[0].k;
        w->raw.assign(source, source + g);
        // Order later runs after any writes already queued by the caller.
        hip_check(hipEventCreateWithFlags(&w->ready, hipEventDisableTiming));
        hip_check(hipEventRecord(w->ready, reinterpret_cast<hipStream_t>(opaque_stream)));
        *result           = w.release();
    });
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_weights_query_v1(ck_tile_int8_nt_cached_weights_handle_v1 w,
                                        ck_tile_int8_nt_cached_weights_info_v1* info)
{
    return api([&] {
        require(w, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null weights handle");
        header(info);
        current_device_is(w->device);
        ck_tile_int8_nt_cached_weights_info_v1 value{};
        value.struct_size = sizeof(value);
        value.abi_version = 1;
        value.group_count = w->g;
        value.n           = w->n;
        value.k           = w->k;
        value.format         = ck_tile::int8_weight_formats::Raw;
        *info                    = value;
    });
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_weights_destroy_v1(ck_tile_int8_nt_cached_weights_handle_v1 w)
{
    return api([&] {
        if(!w)
            return;
        current_device_is(w->device);
        require(w->plans.load() == 0,
                CK_TILE_INT8_CACHED_BUSY_V1,
                "cached weights still have live plans");
        delete w;
    });
}

namespace {
#ifdef CK_INT8_CACHED_TUNING
struct TuningSelection
{
    std::vector<ck_tile::int8_nt_tuning::Candidate>* list;
    unsigned route, legacy;
};
#endif

std::unique_ptr<ck_tile_int8_nt_cached_plan_impl>
make_plan(ck_tile_int8_nt_cached_weights_handle_v1 w,
          const ck_tile_int8_nt_cached_group_desc_v2* groups,
          int32_t g,
          const ck_tile::Int8NtScaleArgs* scales
#ifdef CK_INT8_CACHED_TUNING
          ,
          const TuningSelection* tuning = nullptr
#endif
)
{
    const unsigned output_type = groups[0].output_type;
    int max_m                  = 0;
    int64_t total_m            = 0;
    bool uniform = true, empty = false, native = true, raw_transposed = true;
    bool dense_scalar_a       = true;
    bool dense_channel_scales = true;
    for(int i = 0; i < g; ++i)
    {
        const auto& a = groups[i];
        require(a.m >= 0 && a.m <= 1048560 && a.a_row_stride >= w->k && a.a_row_stride % 16 == 0 &&
                    a.c_row_stride >= w->n && a.c_row_stride % 4 == 0 &&
                    (a.m == 0 || (a.a && a.c && aligned(a.a, 16) && aligned(a.c, 16))),
                CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                "invalid INT8 activation or 16-bit output descriptor");
        require(int64_t(a.m) * a.a_row_stride <= std::numeric_limits<int32_t>::max() &&
                    int64_t(a.m) * a.c_row_stride <= std::numeric_limits<int32_t>::max(),
                CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                "plan matrix span exceeds supported domain");
        uniform &= a.m == groups[0].m;
        empty |= a.m == 0;
        max_m = std::max(max_m, a.m);
        total_m += a.m;
        native &= a.m > 0 && (a.m % 256 == 0 || a.m == 128 || a.m == 64 || a.m == 32) &&
                  int64_t(256) * a.a_row_stride <= std::numeric_limits<int32_t>::max();
        raw_transposed &= a.a_row_stride < 16384;
        const auto& scale = scales[i];
        // The default policy may select paired WASP even when its epilogue
        // needs the generic writeback for a scalar B scale.
        const bool scalar_a = a.c_row_stride == w->n && scale.a_stride == 0;
        dense_scalar_a &= scalar_a;
        dense_channel_scales &= scalar_a && a.a_row_stride == w->k &&
                                w->raw[i].row_stride == w->k && scale.a && scale.b &&
                                scale.b_stride == 1;
    }
    ck_tile::GroupedGemmProblem summary{};
    summary.arch        = ck_tile::GroupedGemmArch::gfx936;
    summary.layout      = ck_tile::GroupedGemmLayout::nt;
    summary.group_count = g;
    summary.m = uniform ? ck_tile::GroupedGemmDimensionSummary::Common(std::max(groups[0].m, 1))
                        : ck_tile::GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(
                              std::max(total_m, int64_t(1)));
    summary.n = ck_tile::GroupedGemmDimensionSummary::Common(w->n);
    summary.k = ck_tile::GroupedGemmDimensionSummary::Common(w->k);
    summary.may_have_empty_groups    = empty;
    summary.group_lengths_are_ragged = !uniform;
    require(ck_tile::is_valid_grouped_gemm_int8_nt_problem(summary),
            CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
            "unsupported INT8 NT shape summary");
    auto p              = std::make_unique<ck_tile_int8_nt_cached_plan_impl>(w, output_type);
    p->g                = g;
    p->max_m            = max_m;
    p->launch_m         = std::max(max_m, 1);
    p->n                = w->n;
    p->k                = w->k;
    const bool native_domain =
        uniform && !empty && native && w->n % 256 == 0 && w->k % 128 == 0 && w->k <= 16384;
    ck_tile::Int8CachedPlanRouteDomain domain{};
    domain.native_wasp          = native_domain;
    domain.uniform_nonempty     = uniform && !empty;
    domain.raw_transposed       = raw_transposed;
    domain.dense_scalar_a       = dense_scalar_a;
    domain.dense_channel_scales = dense_channel_scales;
    domain.fp16_output          = output_type == ck_tile::int8_output_types::Fp16;
    const auto selection =
        ck_tile::int8_cached_select_plan(summary, max_m, w->compute_units, domain);
    auto route         = selection.route;
    p->legacy_instance = static_cast<unsigned>(selection.legacy_instance);
#ifdef CK_INT8_CACHED_TUNING
    if(tuning)
    {
        require(uniform && !empty,
                CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                "tuning requires uniform nonempty scaled groups");
        ck_tile::int8_nt_tuning::Domain domain{max_m, w->n, w->k, 0, 0, true};
        for(int i = 0; i < g; ++i)
        {
            domain.max_a_stride = std::max(domain.max_a_stride, groups[i].a_row_stride);
            domain.max_b_stride = std::max(domain.max_b_stride, w->raw[i].row_stride);
            // Tuning requires the vector-store epilogue. The default policy
            // can also use the generic epilogue of a paired WASP route.
            const auto& scale = scales[i];
            domain.paired &= groups[i].c_row_stride % 8 == 0 && scale.a_stride == 0 &&
                             (!scale.b || scale.b_stride == 1);
        }
        auto candidates = ck_tile::int8_nt_tuning::enumerate(domain, output_type);
        if(tuning->list)
        {
            *tuning->list = std::move(candidates);
            return nullptr;
        }
        const auto it = std::find_if(candidates.begin(), candidates.end(), [&](const auto& entry) {
            return entry.route == tuning->route && entry.legacy_instance == tuning->legacy;
        });
        require(it != candidates.end(),
                CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                "tuning candidate is not legal for these descriptors");
        route              = it->route;
        p->legacy_instance = it->legacy_instance;
    }
#endif
    p->instance_key = route == ck_tile::int8_routes::LegacyRaw
                          ? p->legacy_instance
                          : ck_tile::int8_instance_ids::from_route(route, max_m);
    std::vector<KArg> args;
    args.reserve(g);
    for(int i = 0; i < g; ++i)
    {
        const auto& a = groups[i];
        const auto& b = w->raw[i];
        ck_tile::UniversalGemmKernelArgs<1, 1, 0> arg{{a.a},
                                                      {b.data},
                                                      {},
                                                      a.c,
                                                      a.m,
                                                      w->n,
                                                      w->k,
                                                      {a.a_row_stride},
                                                      {b.row_stride},
                                                      {},
                                                      a.c_row_stride,
                                                      1};
        args.emplace_back(std::move(arg));
    }
    hip_check(hipMalloc(&p->args, sizeof(KArg) * g));
    hip_check(hipMemcpy(p->args, args.data(), sizeof(KArg) * g, hipMemcpyHostToDevice));
    return p;
}
} // namespace

extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_plan_create_v1(ck_tile_int8_nt_cached_weights_handle_v1,
                                      const ck_tile_int8_nt_cached_group_desc_v1*,
                                      int32_t,
                                      ck_tile_int8_nt_cached_plan_handle_v1* result)
{
    return api([&] {
        require(result != nullptr, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null plan result");
        *result = nullptr;
        throw ApiError(CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                       "INT32 output is unsupported; use plan_create_v2 for BF16/FP16");
    });
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_plan_query_v1(ck_tile_int8_nt_cached_plan_handle_v1 p,
                                     ck_tile_int8_nt_cached_plan_info_v1* info)
{
    return api([&] {
        require(p, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null plan handle");
        header(info);
        current_device_is(p->weights->device);
        ck_tile_int8_nt_cached_plan_info_v1 value{};
        value.struct_size         = sizeof(value);
        value.abi_version         = 1;
        value.group_count         = p->g;
        value.max_m               = p->max_m;
        value.n                   = p->n;
        value.k                   = p->k;
        value.route               = ck_tile::int8_instance_ids::route_for(p->instance_key);
        value.legacy_instance     = p->legacy_instance;
        *info                     = value;
    });
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_run_v1(ck_tile_int8_nt_cached_plan_handle_v1 p, void* opaque_stream)
{
    return api([&] {
        require(p, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null plan handle");
        current_device_is(p->weights->device);
        std::lock_guard<std::mutex> lock(p->mutex);
        const auto stream = reinterpret_cast<hipStream_t>(opaque_stream);
        if(std::find(p->streams.begin(), p->streams.end(), stream) == p->streams.end())
        {
            hip_check(hipStreamWaitEvent(stream, p->weights->ready, 0));
            p->streams.push_back(stream);
        }
        const int status = ck_tile_int8_scaled_launch_private(
            p->output_type, p->instance_key, p->args, p->scales, p->g, p->launch_m, p->n, stream);
        require(
            status == 0, CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1, "INT8 cached plan launch failed");
    });
}
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_plan_destroy_v1(ck_tile_int8_nt_cached_plan_handle_v1 p)
{
    return api([&] {
        if(!p)
            return;
        current_device_is(p->weights->device);
        delete p;
    });
}

namespace {
int create_plan_v2(ck_tile_int8_nt_cached_weights_handle_v1 w,
                   const ck_tile_int8_nt_cached_group_desc_v2* groups,
                   int32_t g,
                   ck_tile_int8_nt_cached_plan_handle_v1* result
#ifdef CK_INT8_CACHED_TUNING
                   ,
                   const TuningSelection* tuning = nullptr
#endif
)
{
    return api([&] {
        require(result, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "null plan result");
        *result = nullptr;
        require(w && groups && g == w->g,
                CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                "invalid v2 groups/weights");
        current_device_is(w->device);
        std::vector<ck_tile::Int8NtScaleArgs> scales(g);
        for(int i = 0; i < g; ++i)
        {
            const auto& a = groups[i];
            require(a.struct_size == sizeof(a),
                    CK_TILE_INT8_CACHED_BAD_STRUCT_SIZE_V1,
                    "wrong v2 group size");
            require(a.abi_version == 2,
                    CK_TILE_INT8_CACHED_BAD_VERSION_V1,
                    "v2 group requires abi_version2");
            require(a.reserved[0] == 0 && a.reserved[1] == 0,
                    CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                    "nonzero v2 reserved field");
            require(a.output_type != CK_TILE_INT8_OUTPUT_INT32_V2,
                    CK_TILE_INT8_CACHED_UNSUPPORTED_V1,
                    "INT32 output is unsupported; use BF16 or FP16");
            require((a.output_type == CK_TILE_INT8_OUTPUT_BF16_V2 ||
                     a.output_type == CK_TILE_INT8_OUTPUT_FP16_V2) &&
                        a.output_type == groups[0].output_type,
                    CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                    "invalid/mixed output type");
            require(a.scale_a_mode <= 1 && a.scale_b_mode <= 1,
                    CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                    "unsupported scale mode");
            require((!a.scale_a || aligned(a.scale_a, alignof(float))) &&
                        (!a.scale_b || aligned(a.scale_b, alignof(float))),
                    CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,
                    "unaligned FP32 scale");
            scales[i] = {a.scale_a,
                         a.scale_b,
                         static_cast<int32_t>(a.scale_a_mode),
                         static_cast<int32_t>(a.scale_b_mode)};
        }
        auto p = make_plan(w,
                           groups,
                           g,
                           scales.data()
#ifdef CK_INT8_CACHED_TUNING
                               ,
                           tuning
#endif
        );
#ifdef CK_INT8_CACHED_TUNING
        if(!p) // Enumeration completes before allocating GPU plan buffers.
            return;
#endif
        hip_check(hipMalloc(reinterpret_cast<void**>(&p->scales), sizeof(scales[0]) * g));
        hip_check(
            hipMemcpy(p->scales, scales.data(), sizeof(scales[0]) * g, hipMemcpyHostToDevice));
        *result = p.release();
    });
}
} // namespace

extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_cached_plan_create_v2(ck_tile_int8_nt_cached_weights_handle_v1 w,
                                      const ck_tile_int8_nt_cached_group_desc_v2* groups,
                                      int32_t g,
                                      ck_tile_int8_nt_cached_plan_handle_v1* result)
{
    return create_plan_v2(w, groups, g, result);
}

#ifdef CK_INT8_CACHED_TUNING
extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_tuning_list_v1(ck_tile_int8_nt_cached_weights_handle_v1 w,
                               const ck_tile_int8_nt_cached_group_desc_v2* groups,
                               int32_t g,
                               ck_tile_int8_nt_tuning_candidate_v1* buffer,
                               size_t capacity,
                               size_t* count)
{
    if(!count || (!buffer && capacity))
        return api([&] {
            require(false, CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1, "invalid candidate buffer");
        });
    *count = 0;
    std::vector<ck_tile_int8_nt_tuning_candidate_v1> candidates;
    ck_tile_int8_nt_cached_plan_handle_v1 unused = nullptr;
    const TuningSelection selection{&candidates, 0, 0};
    const int status = create_plan_v2(w, groups, g, &unused, &selection);
    if(status != CK_TILE_INT8_CACHED_SUCCESS_V1)
        return status;
    *count = candidates.size();
    return api([&] {
        if(buffer)
        {
            require(capacity >= candidates.size(),
                    CK_TILE_INT8_CACHED_BAD_STRUCT_SIZE_V1,
                    "candidate buffer is too small");
            std::copy(candidates.begin(), candidates.end(), buffer);
        }
    });
}

extern "C" CK_TILE_INT8_CACHED_API int
ck_tile_int8_nt_tuning_plan_create_v1(ck_tile_int8_nt_cached_weights_handle_v1 w,
                                      const ck_tile_int8_nt_cached_group_desc_v2* groups,
                                      int32_t g,
                                      uint32_t route,
                                      uint32_t legacy_instance,
                                      ck_tile_int8_nt_cached_plan_handle_v1* result)
{
    const TuningSelection selection{nullptr, route, legacy_instance};
    return create_plan_v2(w, groups, g, result, &selection);
}
#endif
