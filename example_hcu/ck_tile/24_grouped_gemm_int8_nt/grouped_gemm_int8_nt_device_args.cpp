// SPDX-License-Identifier: MIT

#include "include/grouped_gemm_int8_nt_device_args.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>

#include "instances/grouped_int8_nt_selector.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"

extern "C" int ck_tile_int8_scaled_launch_private(
    unsigned, unsigned, unsigned, const void*, const void*, int, int, int, hipStream_t);
extern "C" int ck_tile_int8_cached_pack_private(
    const ck_tile_hcu_grouped_gemm_int8_pack_arg_v2*, int, int, int, hipStream_t);

namespace {

using Problem = ck_tile_hcu_grouped_gemm_int8_problem_v2;
using Selection = ck_tile_hcu_grouped_gemm_selection_v1;
using Candidate = ck_tile_hcu_grouped_gemm_int8_candidate_info_v2;
using Launch = ck_tile_hcu_grouped_gemm_int8_launch_v2;

constexpr int UnsupportedArch = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_ARCH_V1;

constexpr int Success = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
constexpr int Invalid = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INVALID_ARGUMENT_V1;
constexpr int Unsupported = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
constexpr int InternalError = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INTERNAL_ERROR_V1;
constexpr auto ScaledAbi = CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION;

template <typename T>
int validate(const T* argument, uint32_t version)
{
    if(argument == nullptr)
        return Invalid;
    if(argument->struct_size != sizeof(T))
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_STRUCT_SIZE_V1;
    if(argument->abi_version != version)
        return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_VERSION_V1;
    if(!std::all_of(std::begin(argument->reserved), std::end(argument->reserved),
                    [](uint64_t reserved) { return reserved == 0; }))
        return Invalid;
    return Success;
}

struct Instance
{
    uint32_t id;
    int32_t output_type;
    unsigned route;
    unsigned legacy;
    const char* name;
};

// The ranges are disjoint from v1 and encode output type, then the compiled
// raw specialization or scaled route. K-parallel has two row specializations.
constexpr uint32_t Base = 0x01000000;
constexpr uint32_t Fp16 = 0x00100000;
constexpr uint32_t Fast = 0x00010000;
constexpr uint32_t KParallelTwoRows = 0x00020000;
constexpr uint32_t KeyMask = 0x0000ffff;

constexpr const char* route_name(unsigned route)
{
    switch(route)
    {
#define CK_INT8_ROUTE(id, abi, kind, weights, ...) case ck_tile::int8_routes::id: return #id;
#include "instances/grouped_int8_nt_routes.inc"
#undef CK_INT8_ROUTE
    default: return nullptr;
    }
}

constexpr const char* legacy_name(unsigned id)
{
    switch(id)
    {
#define CK_INT8_LEGACY_TILE(name, ...) case static_cast<unsigned>(ck_tile::GroupedGemmInstanceId::name): return #name;
#define CK_INT8_LEGACY_WASP(name, ...) CK_INT8_LEGACY_TILE(name)
#define CK_INT8_LEGACY_DIRECT(name) CK_INT8_LEGACY_TILE(name)
#define CK_INT8_LEGACY_DIRECT_EPI(name, ...) CK_INT8_LEGACY_TILE(name)
#include "instances/grouped_int8_nt_legacy_instances.inc"
#undef CK_INT8_LEGACY_DIRECT_EPI
#undef CK_INT8_LEGACY_DIRECT
#undef CK_INT8_LEGACY_WASP
#undef CK_INT8_LEGACY_TILE
    default: return nullptr;
    }
}

Instance find_instance(uint32_t id)
{
    using namespace ck_tile::int8_routes;
    constexpr uint32_t ValidBits = Base | Fp16 | Fast | KParallelTwoRows | KeyMask;
    if((id & Base) == 0 || (id & ~ValidBits) != 0)
        return {};
    const auto key = id & KeyMask;
    const auto route = (id & Fast) ? key : 0;
    if((id & KParallelTwoRows) && route != KParallelDecode)
        return {};
    const char* name = (id & Fast) ? route_name(route) : legacy_name(key);
    if(name == nullptr)
        return {};
    return {id,
            (id & Fp16) ? CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1
                        : CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1,
            route,
            route == 0 ? key : 0,
            name};
}

void copy_instance_name(char* out, const Instance& instance)
{
    const bool bf16 = instance.output_type == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1;
    const char* rows = "";
    if(instance.route == ck_tile::int8_routes::KParallelDecode)
        rows = (instance.id & KParallelTwoRows) ? "_m2" : "_m1";
    std::snprintf(out, CK_TILE_HCU_GROUPED_GEMM_INSTANCE_NAME_CAPACITY,
                  "gfx936_int8_nt_%s_%s%s", bf16 ? "bf16" : "fp16", instance.name, rows);
}

int select(const Problem* p, Instance& instance)
{
    const int status = validate(p, ScaledAbi);
    if(status != Success)
        return status;

    constexpr unsigned Empty = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1;
    constexpr unsigned Ragged = CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
    constexpr unsigned Dense = CK_TILE_HCU_GROUPED_GEMM_INT8_DENSE_CHANNEL_SCALE_V2;
    constexpr unsigned Packed = CK_TILE_HCU_GROUPED_GEMM_INT8_PACKED_B_AVAILABLE_V2;
    if((p->problem_flags & ~(Empty | Ragged)) || (p->int8_flags & ~(Dense | Packed)) ||
       p->m.reserved != 0)
        return Invalid;
    if(p->output_type != CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1 &&
       p->output_type != CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1)
        return Unsupported;

    const bool common = p->m.kind == CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1;
    const bool capacity =
        p->m.kind == CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1;
    const bool uniform = common && (p->problem_flags & (Empty | Ragged)) == 0;
    const bool dense = (p->int8_flags & Dense) != 0;
    const bool can_pack = (p->int8_flags & Packed) != 0;
    if((!common && !capacity) || (common && (p->m.extent != p->max_m ||
                                            (p->problem_flags & Ragged))) ||
       p->max_m <= 0 || p->max_m > p->m.extent || p->m.extent <= 0 ||
       p->max_stride_a < p->k || p->max_stride_b < p->k || p->max_stride_c < p->n ||
       p->max_stride_a % 16 != 0 || p->max_stride_b % 16 != 0 || p->max_stride_c % 4 != 0 ||
       (dense && (p->max_stride_a != p->k || p->max_stride_b != p->k ||
                  p->max_stride_c != p->n)))
        return Invalid;

    constexpr auto MaxSpan = std::numeric_limits<int32_t>::max();
    if(int64_t(p->max_m) * p->max_stride_a > MaxSpan ||
       int64_t(p->max_m) * p->max_stride_c > MaxSpan ||
       int64_t(p->n) * p->max_stride_b > MaxSpan)
        return Unsupported;

    if(ck_tile::get_hcu_target_enum() != ck_tile::hcu_target_enum::gfx936)
        return UnsupportedArch;
    ck_tile::GroupedGemmProblem legacy{};
    legacy.arch        = ck_tile::GroupedGemmArch::gfx936;
    legacy.data_type   = ck_tile::GroupedGemmDataType::int8_int32;
    legacy.layout      = ck_tile::GroupedGemmLayout::nt;
    legacy.group_count = p->group_count;
    legacy.m = common ? ck_tile::GroupedGemmDimensionSummary::Common(p->m.extent)
                      : ck_tile::GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(p->m.extent);
    legacy.n = ck_tile::GroupedGemmDimensionSummary::Common(p->n);
    legacy.k = ck_tile::GroupedGemmDimensionSummary::Common(p->k);
    legacy.may_have_empty_groups = (p->problem_flags & Empty) != 0;
    legacy.group_lengths_are_ragged = (p->problem_flags & Ragged) != 0;
    const auto fallback = ck_tile::select_grouped_gemm_candidate(legacy);
    if(!fallback.supported())
        return Unsupported;

    const auto m = p->max_m;
    const bool native = uniform && (m % 256 == 0 || m == 128 || m == 64 || m == 32) &&
                        p->n % 256 == 0 && p->k % 128 == 0 && p->k < 16384 &&
                        int64_t(256) * p->max_stride_a <= MaxSpan;
    const bool packed =
        can_pack && p->n % 256 == 0 && p->k % 128 == 0 && p->k >= 512 && p->k < 16384;
    const auto route = ck_tile::int8_cached_select_plan_route(
        m, p->n, p->k, p->group_count, packed, ck_tile::get_num_cu(), true,
        {native, uniform, p->max_stride_a < 16384, dense, dense});
    const auto id = Base |
                    (p->output_type == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1 ? Fp16 : 0) |
                    (route == ck_tile::int8_routes::LegacyRaw
                         ? static_cast<uint32_t>(fallback.instance_id)
                         : Fast | static_cast<uint32_t>(route)) |
                    (route == ck_tile::int8_routes::KParallelDecode && m == 2 ? KParallelTwoRows : 0);
    instance = find_instance(id);
    return instance.id == 0 ? Unsupported : Success;
}

int write_selection(const Instance& instance, Selection* out)
{
    const int status = validate(out, CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION);
    if(status != Success)
        return status;
    *out = {};
    out->struct_size = sizeof(*out);
    out->abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    out->instance_id = instance.id;
    out->reason = CK_TILE_HCU_GROUPED_GEMM_SELECTION_INT8_NT_V1;
    copy_instance_name(out->instance_name, instance);
    return Success;
}

} // namespace

extern "C" int ck_tile_hcu_grouped_gemm_select_int8_nt_device_args_v2(
    const Problem* problem, Selection* output)
{
    try
    {
        Instance instance{};
        const int status = select(problem, instance);
        return status == Success ? write_selection(instance, output) : status;
    }
    catch(...)
    {
        return InternalError;
    }
}

extern "C" int ck_tile_hcu_grouped_gemm_query_int8_nt_instance_v2(uint32_t id, Candidate* out)
{
    const int status = validate(out, ScaledAbi);
    if(status != Success)
        return status;
    const auto instance = find_instance(id);
    if(instance.id == 0)
        return Unsupported;
    *out = {};
    out->struct_size = sizeof(*out);
    out->abi_version = ScaledAbi;
    out->instance_id = id;
    out->architecture_mask = CK_TILE_HCU_GROUPED_GEMM_ARCHITECTURE_MASK_GFX936_V1;
    out->layout_mask = CK_TILE_HCU_GROUPED_GEMM_LAYOUT_MASK_NT_V1;
    out->output_type = instance.output_type;
    if(ck_tile::int8_routes::uses_packed_weights(instance.route))
        out->contract_flags = CK_TILE_HCU_GROUPED_GEMM_INT8_PACKED_B_V2;
    copy_instance_name(out->canonical_name, instance);
    return Success;
}

extern "C" int ck_tile_hcu_grouped_gemm_run_int8_nt_device_args_v2(
    const Problem* problem, const Launch* launch, Selection* output)
{
    try
    {
        int status = validate(launch, ScaledAbi);
        if(status != Success)
            return status;
        if(launch->device_args == nullptr || launch->device_scales == nullptr)
            return Invalid;
        if(launch->num_cu != 0)
            return Unsupported;

        Instance instance{};
        status = select(problem, instance);
        if(status != Success)
            return status;
        if(launch->expected_instance_id != 0 && launch->expected_instance_id != instance.id)
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1;
        if(output != nullptr)
        {
            status = write_selection(instance, output);
            if(status != Success)
                return status;
        }
        const auto output_type = instance.output_type == CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1
                                     ? ck_tile::int8_output_types::Bf16
                                     : ck_tile::int8_output_types::Fp16;
        status = ck_tile_int8_scaled_launch_private(output_type,
                                                   instance.route,
                                                   instance.legacy,
                                                   launch->device_args,
                                                   launch->device_scales,
                                                   problem->group_count,
                                                   problem->max_m,
                                                   problem->n,
                                                   static_cast<hipStream_t>(launch->stream));
        return status == 0 ? Success : InternalError;
    }
    catch(...)
    {
        return InternalError;
    }
}

extern "C" int ck_tile_hcu_grouped_gemm_pack_int8_nt_weights_v2(
    const ck_tile_hcu_grouped_gemm_int8_pack_arg_v2* args, int32_t g, int32_t n, int32_t k, void* stream)
{
    try
    {
        if(args == nullptr || g <= 0 || g > 65535 || n <= 0 || k < 512 || k >= 16384 ||
           n % 256 != 0 || k % 128 != 0 || int64_t(n) * k > std::numeric_limits<int32_t>::max())
            return Invalid;
        if(ck_tile::get_hcu_target_enum() != ck_tile::hcu_target_enum::gfx936)
            return UnsupportedArch;
        return ck_tile_int8_cached_pack_private(args, g, n, k, static_cast<hipStream_t>(stream))
                       == 0
                   ? Success
                   : InternalError;
    }
    catch(...)
    {
        return InternalError;
    }
}
