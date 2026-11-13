// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace ck_tile {

enum class GroupedGemmArch
{
    unsupported,
    gfx936,
    gfx938,
    gfx946,
};

enum class GroupedGemmDataType
{
    unsupported,
    fp16,
    bf16,
};

enum class GroupedGemmLayout
{
    unsupported,
    nt,
    nn,
    tn,
};

// A dimension is either common to every group or represented by device-side
// per-group lengths backed by one physical allocation. The selector may use
// the capacity as a conservative host-visible summary, but must not treat it
// as an individual group's logical length.
enum class GroupedGemmDimensionKind
{
    common,
    device_lengths_with_capacity,
};

struct GroupedGemmDimensionSummary
{
    GroupedGemmDimensionKind kind = GroupedGemmDimensionKind::common;
    std::int64_t extent           = 0;

    static constexpr GroupedGemmDimensionSummary Common(std::int64_t value)
    {
        return {GroupedGemmDimensionKind::common, value};
    }

    static constexpr GroupedGemmDimensionSummary DeviceLengthsWithCapacity(
        std::int64_t capacity)
    {
        return {GroupedGemmDimensionKind::device_lengths_with_capacity, capacity};
    }
};

struct GroupedGemmProblem
{
    GroupedGemmArch arch           = GroupedGemmArch::unsupported;
    GroupedGemmDataType data_type  = GroupedGemmDataType::unsupported;
    GroupedGemmLayout layout       = GroupedGemmLayout::unsupported;
    std::int32_t group_count       = 0;
    GroupedGemmDimensionSummary m  = {};
    GroupedGemmDimensionSummary n  = {};
    GroupedGemmDimensionSummary k  = {};
    bool may_have_empty_groups     = false;
    bool group_lengths_are_ragged  = false;
};

// Policy is an explicit input so selection is deterministic and testable.
// Environment-variable handling and hardware detection belong to API adapters.
struct GroupedGemmSelectorPolicy
{
    bool enable_bw_family                 = true;
    bool enable_gfx936_v3                 = true;
    bool enable_gfx936_dsreadm            = true;
    bool enable_gfx938_mls                = true;
    bool enable_gfx936_bf16_nn_promoted   = true;
    bool enable_bf16_tn_logical_k_tail    = true;
    bool enable_bw_selected               = true;
    bool enable_gfx938_mls_large          = true;
    // P1 production defaults after architecture-local correctness and full G392 replay.
    bool enable_p1_family_a_v3_128_gfx938 = true;
    bool enable_p1_family_a_v3_128_gfx936 = true;
    bool enable_p1_family_b_v6_m32_mle8_gfx936 = true;
    // Production default after independent gfx936 focused/full316 validation. Callers can
    // still set this field false to roll the narrow fixed-M route back to V4 64x128 M01=1.
    bool enable_p2_bf16_g8_v4_64_m01_4_gfx936 = true;
    // Production default after independent gfx936 correctness/focused validation. Callers can
    // still set this field false to roll the exact FC2 NN route back to bw_family_selected.
    bool enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = true;
    // Production default after dual-session focused/full316 validation. Callers can
    // still set this field false to roll the narrow gfx938 FC1 route back to MLS.
    bool enable_p2_bf16_g8_gfx938_v5_fc1_route = true;
    // Production default after independent gfx936/gfx938 correctness and focused fallback
    // comparisons. Callers can still set this field false for an explicit V4 rollback.
    bool enable_p3_fp16_tn_m1024_k4096_route = true;
    // v7 BF16 NN / TN (the uniform Direct-to-LDS kernels as device-args
    // instances; see instances/grouped_gemm_v7_v8_bf16_{nn,tn}_device_args.cpp).
    //
    // OPT-IN, unlike every route above: these are the first landings of the v7
    // uniform-pipeline kernels as CK instances, and the production NN/TN routes
    // must not move until they are measured on the real shapes. The adapter also
    // honours CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_V8 so the A/B does not
    // need a rebuild; the disable side of those switches wins.
    bool enable_bf16_v7_nn = false;
    bool enable_bf16_v8_tn = false;
    // Inside the v7 NN route: K past kGroupedGemmV7NnM01_8KThreshold takes
    // 5002 (the same kernel with the M01 = 8 tile walk). Set false to keep
    // every v7 NN problem on 5000.
    bool enable_bf16_v7_nn_m01_8 = true;
    // Experimental small-M candidates; default routes stay unchanged.
    bool enable_bf16_v7_nn_small_m = false;
    std::int32_t bf16_v7_nn_small_m_tile_n = 128;
    bool enable_bf16_v7_nn_a_noslc = false; // Independent cache-policy experiment.
    // P6f diagnostic factor: 0 keeps the old policy; 1/5/8 set a tile walk.
    std::int32_t bf16_v7_nn_cache_walk = 0;
    bool enable_bf16_v7_nn_cache_grid = false;
    bool enable_bf16_v7_nn_tail128 = false;
    std::int32_t bf16_v7_nn_tail_tile_n = 128;
    // 0=off, 1=K16 drain, 2=K32 N128, 3=square128, 4=K32 N64.
    std::int32_t bf16_v7_nn_small_m_experiment = 0;
    // Experimental archived A-register/B-private-LDS Depth8, narrow measured scope.
    bool enable_bf16_nn_decode_depth8 = false;
    bool enable_bf16_nn_decode_wave = false; // Independent, opt-in decode architecture.
    bool enable_bf16_nn_decode_mainline = false;
    std::int32_t bf16_nn_decode_mainline_variant = 0;
    bool enable_bf16_nn_decode_short = false;
    std::int32_t bf16_nn_decode_short_variant = 0;
};

// Group-count bounds of the v7 instances. The kernels read the group table
// through the device kargs pointer, so G is bounded by what each kernel's
// group lookup was validated for, not by a table size. NN (5000) locates the
// group in O(1) for a common M and by a 64-wide wave scan otherwise; TN (5001)
// keeps the scalar scan and has only been validated up to 16 groups.
constexpr std::int32_t kGroupedGemmV7NnMaxGroups = 1024;
constexpr std::int32_t kGroupedGemmV8TnMaxGroups = 16;

// The v7 kernels address their tiles with index_t block ids and tile prefix
// sums. The gate leaves headroom below INT32_MAX for the launcher's round-up to
// whole CTA slots.
constexpr std::int64_t kGroupedGemmV7MaxTiles = std::int64_t{1} << 30;

// v7 NN tile walk. Past this K the row-major walk of 5000 loses L2 reuse of the
// A/B panels (the CTAs in flight drift apart in k): measured on gfx936,
// M8192 N4096 G8, L2 hit 84% at K4096 vs 51% at K7168. 5002's M01 = 8 walk
// was faster at 39 of the 40 K>4096 grid points (K 4352..6144, N 3072..7168,
// M 2048/8192), up to 2.3x; at K4096 it is 2-4% slower for N <= 4096, so the
// threshold is exclusive. The one K>4096 loss was M8192 N3072 K6144 (+3.8%).
constexpr std::int64_t kGroupedGemmV7NnM01_8KThreshold = 4096;

enum class GroupedGemmInstanceId
{
    // Values are stable identity codes. Add new instances to a family range;
    // never renumber an existing entry.
    none                                            = 0,
    bw_family_selected                              = 1000,
    bw_family_bf16_nn_fixed_srd_lds8               = 1001,
    bw_family_bf16_nn_fast_vmem3                    = 1002,
    bw_family_blas_transposed                       = 1003,
    bw_family_logical_k_tail                        = 1004,
    gfx936_v3_256_m_only_padding                    = 2000,
    gfx936_v3_dsreadm_backward_m_only_padding       = 2001,
    gfx936_v3_default_backward_m_only_padding       = 2002,
    gfx938_mls_large_256                            = 3000,
    gfx938_mls_small_128                            = 3001,
    v4_128_m_only_padding                          = 4000,
    v4_128_full_padding                            = 4001,
    v4_64_nonpadding                               = 4002,
    v4_64_full_padding                             = 4003,
    v3_128_m4_full_padding                          = 4009,
    v3_128_m4_full_padding_gfx936                   = 4010,
    v4_64_m01_4_gfx936                              = 4011,
    v6_m32_nonpadding_mle8_family_b_gfx936          = 4104,
    // v7 family (the uniform Direct-to-LDS NN/TN kernels as device-args
    // instances).
    v7_bf16_nn                                     = 5000,
    v8_bf16_tn                                     = 5001,
    v7_bf16_nn_m01_8                               = 5002,
    v7_bf16_nn_small_m128                          = 5003,
    v7_bf16_nn_small_m64                           = 5004,
    v7_bf16_nn_a_noslc                             = 5005,
    v7_bf16_nn_a_noslc_m01_8                       = 5006,
    v7_bf16_nn_small_m_drain = 5007,
    v7_bf16_nn_small_m_k32 = 5008,
    v7_bf16_nn_small_m_square128 = 5009,
    v7_bf16_nn_small_m_k32n64 = 5010,
    v7_bf16_nn_cache_m01_5 = 5011,
    v7_bf16_nn_cache_noslc_m01_5 = 5012,
    v7_bf16_nn_tail128 = 5013,
    v7_bf16_nn_small_m_k32_steady = 5014,
    v7_bf16_nn_small_m_k16_steady = 5015,
    v7_bf16_nn_tail256 = 5016,
    v7_bf16_nn_small_m_noslc = 5017,
    bf16_nn_decode_depth8 = 6003,
    bf16_nn_decode_wave = 6000,
    bf16_nn_decode_mainline_d4 = 6004,
    bf16_nn_decode_mainline_d8 = 6005,
    bf16_nn_decode_mainline_serial = 6006,
    bf16_nn_decode_short_n64 = 6007,
    bf16_nn_decode_short_n32 = 6008,
    bf16_nn_decode_shared_k16_d4 = 6009,
    bf16_nn_decode_shared_k16_d8 = 6010,
    bf16_nn_decode_shared_k32_d4 = 6011,
    bf16_nn_decode_short_shared = 6012,
    bf16_nn_decode_direct_main_xor = 6013,
    bf16_nn_decode_direct_main_early = 6014,
    bf16_nn_decode_direct_main_both = 6015,
    bf16_nn_decode_direct_short_xor = 6016,
    bf16_nn_decode_direct_short_early = 6017,
    bf16_nn_decode_direct_short_both = 6018,
    bf16_nn_decode_coop_k16_d4 = 6020,
    bf16_nn_decode_coop_k16_d8 = 6021,
    bf16_nn_decode_coop_k32_d4 = 6022,
    bf16_nn_decode_coop_k32_d5 = 6023,
    bf16_nn_decode_coop_k32_a2_b3 = 6024,
    bf16_nn_decode_coop_k32_d4_w8 = 6025,
    bf16_nn_decode_coop_k32_a2_b3_phase = 6026,
    bf16_nn_decode_coop_k32_d4_w8_phase = 6027,
    bf16_nn_decode_coop_k64_d2 = 6028,
    bf16_nn_decode_coop_k64_d2_w8 = 6029,
    bf16_nn_decode_coop_n256_k32_d2_w8 = 6030,
    bf16_nn_decode_coop_n256_k32_d3_w8 = 6031,
    bf16_nn_decode_coop_n256_b_reuse_d2 = 6032,
    bf16_nn_decode_coop_n256_b_reuse_d3 = 6033,

};

enum class GroupedGemmCandidateFamily
{
    bw_family,
    gfx936_v3,
    gfx938_mls,
    v4,
    v7,
    decode,
};

struct GroupedGemmCandidateIdentity
{
    GroupedGemmInstanceId id;
    std::string_view canonical_suffix;
    GroupedGemmCandidateFamily family;
};

inline constexpr std::array<GroupedGemmCandidateIdentity, 67>
    grouped_gemm_candidate_registry{{
        {GroupedGemmInstanceId::bw_family_selected,
         "bw_family.selected",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8,
         "bw_family.rr_fixed_srd_lds8_m1920_2304_n7168_k4096",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
         "bw_family.rr_fast_vmem3_m1920_2304_n2048_k7168",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_blas_transposed,
         "bw_family.blas_transposed",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::bw_family_logical_k_tail,
         "bw_family.logical_k_tail",
         GroupedGemmCandidateFamily::bw_family},
        {GroupedGemmInstanceId::gfx936_v3_256_m_only_padding,
         "v3.256x256x64.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
         "v3.dsreadm_stage_backward_hybrid.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding,
         "v3.default_backward_hybrid.m_only_padding",
         GroupedGemmCandidateFamily::gfx936_v3},
        {GroupedGemmInstanceId::gfx938_mls_large_256,
         "mls.large.256x256x32",
         GroupedGemmCandidateFamily::gfx938_mls},
        {GroupedGemmInstanceId::gfx938_mls_small_128,
         "mls.small.128x128x32",
         GroupedGemmCandidateFamily::gfx938_mls},
        {GroupedGemmInstanceId::v4_128_m_only_padding,
         "v4.128x128x64.m_only_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_128_full_padding,
         "v4.128x128x64.full_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_nonpadding,
         "v4.64x128x64.nonpadding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_full_padding,
         "v4.64x128x64.full_padding",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v3_128_m4_full_padding,
         "v3.128x128x64.m4.full_padding.single_lds",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936,
         "v3.128x128x64.m4.full_padding.single_lds.gfx936",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v4_64_m01_4_gfx936,
         "v4.64x128x64.nonpadding.m01_4.gfx936",
         GroupedGemmCandidateFamily::v4},
        {GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936,
         "v6.32x128x64.nonpadding.selector_m_le_8.family_b.gfx936",
         GroupedGemmCandidateFamily::v4},
        // v7: 256x256x16 tile, 8 waves (4m x 2n), 4-stage 16 KB LDS ring fed
        // by Direct-to-LDS buffer loads, m32x16 / b64 LDS reads, two-group phase
        // interleave and the LDS-staged dwordx4 epilogue. The suffix names the
        // operand form, because that is what separates the NN entry (only the
        // second operand is [k, feature]) from the TN one (both are).
        {GroupedGemmInstanceId::v7_bf16_nn,
         "v7.nn_b.uniform_d2l.256x256x16",
         GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v8_bf16_tn,
         "v8.tn_ab.uniform_d2l.256x256x16",
         GroupedGemmCandidateFamily::v7},
        // 5000 with the M01 = 8 tile walk (see kGroupedGemmV7NnM01_8KThreshold).
        {GroupedGemmInstanceId::v7_bf16_nn_m01_8,
         "v7.nn_b.uniform_d2l.256x256x16.m01_8",
         GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m128,
         "v7.nn_b.small_m.64x128x16.s4", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m64,
         "v7.nn_b.small_m.64x64x16.s4", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_a_noslc,
         "v7.nn_b.uniform_d2l.256x256x16.a_noslc", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_a_noslc_m01_8,
         "v7.nn_b.uniform_d2l.256x256x16.a_noslc.m01_8", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_drain,
         "v7.nn_b.small_m.64x128x16.drain", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32,
         "v7.nn_b.small_m.64x128x32", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_square128,
         "v7.nn_b.small_m.128x128x16", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64,
         "v7.nn_b.small_m.64x64x32", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_cache_m01_5,
         "v7.nn_b.uniform_d2l.256x256x16.cache_m01_5", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,
         "v7.nn_b.uniform_d2l.256x256x16.cache_noslc_m01_5", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_tail128,
         "v7.nn_b.tail128", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32_steady,
         "v7.nn_b.small_m_k32_steady", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k16_steady,
         "v7.nn_b.small_m_k16_steady", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_tail256,
         "v7.nn_b.tail256", GroupedGemmCandidateFamily::v7},
        {GroupedGemmInstanceId::v7_bf16_nn_small_m_noslc,
         "v7.nn_b.small_m_noslc", GroupedGemmCandidateFamily::v7},

        {GroupedGemmInstanceId::bf16_nn_decode_depth8,
         "decode.nn.a_reg.b_private_lds.16x128x16.s8", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_wave,
         "decode.nn.wave_private.16x128x16.s2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_mainline_d4,
         "decode.nn.full_m.a_reg.b_private_lds.n128.k16.d4.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_mainline_d8,
         "decode.nn.full_m.a_reg.b_private_lds.n128.k16.d8.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_mainline_serial,
         "decode.nn.full_m.a_reg.b_private_lds.n128.k16.d4.serial", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_short_n64,
         "decode.nn.full_m.a_reg.b_private_lds.n64.k16.d2.serial", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_short_n32,
         "decode.nn.full_m.a_reg.b_private_lds.n32.k16.d2.serial", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_shared_k16_d4,
         "decode.nn.full_m.shared_a.private_b.n128.k16.d4.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_shared_k16_d8,
         "decode.nn.full_m.shared_a.private_b.n128.k16.d8.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_shared_k32_d4,
         "decode.nn.full_m.shared_a.private_b.n128.k32.d4.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_short_shared,
         "decode.nn.short.shared_a.private_b.n128.k16.d4.op2", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_main_xor,
         "decode.nn.full_m.shared_a.private_b.n128.k16.d8.xor_a.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_main_early,
         "decode.nn.full_m.shared_a.private_b.n128.k16.d8.early_b.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_main_both,
         "decode.nn.full_m.shared_a.private_b.n128.k16.d8.xor_a.early_b.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_short_xor,
         "decode.nn.short.shared_a.private_b.n128.k16.d4.xor_a.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_short_early,
         "decode.nn.short.shared_a.private_b.n128.k16.d4.early_b.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_direct_short_both,
         "decode.nn.short.shared_a.private_b.n128.k16.d4.xor_a.early_b.direct", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k16_d4,
         "decode.nn.full_m.cooperative_ab.n128.k16.d4.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k16_d8,
         "decode.nn.full_m.cooperative_ab.n128.k16.d8.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4,
         "decode.nn.full_m.cooperative_ab.n128.k32.d4.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d5,
         "decode.nn.full_m.cooperative_ab.n128.k32.d5.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3,
         "decode.nn.full_m.cooperative_ab.n128.k32.a2.b3.w4.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8,
         "decode.nn.full_m.cooperative_ab.n128.k32.d4.w8.op3", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3_phase,
         "decode.nn.full_m.cooperative_ab.n128.k32.a2.b3.w4.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8_phase,
         "decode.nn.full_m.cooperative_ab.n128.k32.d4.w8.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2,
         "decode.nn.full_m.cooperative_ab.n128.k64.d2.w4.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2_w8,
         "decode.nn.full_m.cooperative_ab.n128.k64.d2.w8.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d2_w8,
         "decode.nn.full_m.cooperative_ab.n256.k32.d2.w8.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d3_w8,
         "decode.nn.full_m.cooperative_ab.n256.k32.d3.w8.op3.phase_first", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d2,
         "decode.nn.full_m.cooperative_ab.n256.k32.d2.w8.op3.b_reuse", GroupedGemmCandidateFamily::decode},
        {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d3,
         "decode.nn.full_m.cooperative_ab.n256.k32.d3.w8.op3.b_reuse", GroupedGemmCandidateFamily::decode},

    }};

constexpr const GroupedGemmCandidateIdentity*
find_grouped_gemm_candidate(GroupedGemmInstanceId id)
{
    for(const auto& candidate : grouped_gemm_candidate_registry)
    {
        if(candidate.id == id)
        {
            return &candidate;
        }
    }
    return nullptr;
}

enum class GroupedGemmSelectionReason
{
    invalid_contract,
    bw_family,
    gfx936_v3,
    gfx938_mls,
    v4_aligned,
    v4_bf16_padding,
    p1_family_a_v3_128_gfx938,
    p1_family_a_v3_128_gfx936,
    p1_family_b_v6_m32_mle8_gfx936,
    p2_bf16_g8_v4_64_m01_4_gfx936,
    p2_bf16_g8_gfx936_fc2_nn_fast_vmem3,
    p2_bf16_g8_gfx938_v5_fc1_route,
    p3_fp16_tn_m1024_k4096_route,
    v7_bf16_nn,
    v8_bf16_tn,
    v4_n_aligned,
    v4_fallback,
    // Appended: the reason is reported as its integer value, so existing
    // entries keep their numbers.
    v7_bf16_nn_m01_8,
    v7_bf16_nn_small_m128,
    v7_bf16_nn_small_m64,
    v7_bf16_nn_a_noslc,
    v7_bf16_nn_a_noslc_m01_8,
    v7_bf16_nn_small_m_drain,
    v7_bf16_nn_small_m_k32,
    v7_bf16_nn_small_m_square128,
    v7_bf16_nn_small_m_k32n64,
    v7_bf16_nn_cache_m01_5,
    v7_bf16_nn_cache_noslc_m01_5,
    v7_bf16_nn_tail128,
    v7_bf16_nn_small_m_k32_steady,
    v7_bf16_nn_small_m_k16_steady,
    v7_bf16_nn_tail256,
    v7_bf16_nn_small_m_noslc,
    bf16_nn_decode_wave,
    bf16_nn_decode_depth8 = 6003,
    bf16_nn_decode_mainline = 6004,
    bf16_nn_decode_short = 6007,
};

struct GroupedGemmSelection
{
    GroupedGemmInstanceId instance_id = GroupedGemmInstanceId::none;
    GroupedGemmSelectionReason reason  = GroupedGemmSelectionReason::invalid_contract;

    constexpr bool supported() const { return instance_id != GroupedGemmInstanceId::none; }
};

constexpr bool is_valid_grouped_gemm_problem(const GroupedGemmProblem& problem)
{
    if((problem.arch != GroupedGemmArch::gfx936 && problem.arch != GroupedGemmArch::gfx938 &&
        problem.arch != GroupedGemmArch::gfx946) ||
       (problem.data_type != GroupedGemmDataType::fp16 &&
        problem.data_type != GroupedGemmDataType::bf16) ||
       (problem.layout != GroupedGemmLayout::nt && problem.layout != GroupedGemmLayout::nn &&
        problem.layout != GroupedGemmLayout::tn) ||
       problem.group_count <= 0 || problem.m.extent <= 0 || problem.n.extent <= 0 ||
       problem.k.extent <= 0 || problem.n.kind != GroupedGemmDimensionKind::common)
    {
        return false;
    }

    const int variable_dimensions =
        (problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0) +
        (problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1 : 0);
    return variable_dimensions <= 1;
}

constexpr std::int64_t grouped_gemm_selector_m_extent(const GroupedGemmProblem& problem)
{
    return problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity
               ? problem.m.extent / problem.group_count
               : problem.m.extent;
}

constexpr std::int64_t grouped_gemm_selector_k_extent(const GroupedGemmProblem& problem)
{
    return problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity
               ? problem.k.extent / problem.group_count
               : problem.k.extent;
}

// Upper bound on the v7 (group, m tile, n tile) count, from host-visible
// lengths/capacities only. A common M is every group's length; a device-resident
// M is bounded by its capacity, and each group can round up to its own tile, so
// ceil(capacity / 256) + group_count bounds the summed row tiles. The launcher
// sizes its grid from this same bound (grouped_gemm_v7_uniform_launch.hpp).
constexpr std::int64_t grouped_gemm_v7_tile_bound(const GroupedGemmProblem& problem)
{
    const std::int64_t groups  = problem.group_count > 0 ? problem.group_count : 1;
    const std::int64_t n_tiles = (problem.n.extent + 255) / 256;
    const bool device_m =
        problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity;
    const std::int64_t m_tiles = device_m ? (problem.m.extent + 255) / 256 + groups
                                          : (problem.m.extent + 255) / 256 * groups;
    return m_tiles * n_tiles;
}

// Common-M prototype bound. No device length read or allocation is needed.
constexpr std::int64_t grouped_gemm_v7_small_m_tile_bound(
    const GroupedGemmProblem& problem, std::int32_t tile_n)
{
    if(tile_n != 64 && tile_n != 128)
        return kGroupedGemmV7MaxTiles + 1;
    return ((problem.m.extent + 63) / 64) *
           ((problem.n.extent + tile_n - 1) / tile_n) * problem.group_count;
}

// Experimental geometry is host-visible; no device length inspection.
constexpr std::int64_t grouped_gemm_v7_experiment_tile_bound(
    const GroupedGemmProblem& problem, std::int32_t tile_m, std::int32_t tile_n)
{
    return ((problem.m.extent + tile_m - 1) / tile_m) *
           ((problem.n.extent + tile_n - 1) / tile_n) * problem.group_count;
}

constexpr GroupedGemmSelection select_grouped_gemm_candidate(
    const GroupedGemmProblem& problem,
    const GroupedGemmSelectorPolicy& policy = {})
{
    if(!is_valid_grouped_gemm_problem(problem))
    {
        return {};
    }

    const auto effective_m = grouped_gemm_selector_m_extent(problem);
    const auto effective_k = grouped_gemm_selector_k_extent(problem);
    const auto m           = problem.m.extent;
    const auto n           = problem.n.extent;
    const auto k           = problem.k.extent;
    const bool is_nt       = problem.layout == GroupedGemmLayout::nt;
    const bool is_nn       = problem.layout == GroupedGemmLayout::nn;
    const bool is_tn       = problem.layout == GroupedGemmLayout::tn;
    const bool is_bf16     = problem.data_type == GroupedGemmDataType::bf16;
    const bool is_bf16_tn  = is_bf16 && is_tn;
    const bool has_variable_k =
        problem.k.kind == GroupedGemmDimensionKind::device_lengths_with_capacity;
    // Independent experimental decode routes. Packed/aligned dense NN device
    // descriptors retain the existing caller-owned storage contract. Neither
    // route reads device lengths or adds workspace, synchronization, or an ABI.
    const bool decode_common = problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
        problem.m.kind == GroupedGemmDimensionKind::common && !has_variable_k &&
        !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
        problem.group_count <= 1024 && m >= 1 && m <= 64 && n <= 4096 && n % 128 == 0;
    if(policy.enable_bf16_nn_decode_mainline && decode_common && n >= 1024 &&
       k >= 2048 && k <= 8192 && k % 128 == 0 &&
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, 64, 128) <= kGroupedGemmV7MaxTiles)
    {
        switch(policy.bf16_nn_decode_mainline_variant)
        {
        case 0: return {GroupedGemmInstanceId::bf16_nn_decode_mainline_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 1: return {GroupedGemmInstanceId::bf16_nn_decode_mainline_d8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 2: return {GroupedGemmInstanceId::bf16_nn_decode_mainline_serial,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 3: return {GroupedGemmInstanceId::bf16_nn_decode_shared_k16_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 4: return {GroupedGemmInstanceId::bf16_nn_decode_shared_k16_d8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 5: return {GroupedGemmInstanceId::bf16_nn_decode_shared_k32_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 6: return {GroupedGemmInstanceId::bf16_nn_decode_direct_main_xor,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 7: return {GroupedGemmInstanceId::bf16_nn_decode_direct_main_early,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 8: return {GroupedGemmInstanceId::bf16_nn_decode_direct_main_both,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 9: return {GroupedGemmInstanceId::bf16_nn_decode_coop_k16_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 10: return {GroupedGemmInstanceId::bf16_nn_decode_coop_k16_d8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 11: return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 12: return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d5,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        // Experimental attribute domain; every member still needs independent
        // confirmation, including existing winners and unseen geometries.
        case 13:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 14:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d5,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 15:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_direct_main_xor,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 16:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_direct_main_both,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 17:
            return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3,
                    GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 18:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 19:
            if(m >= 17)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 20:
            return {m <= 16 ? GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3 :
                              GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8,
                    GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 21:
            return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3_phase,
                    GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 22:
            if(m >= 17)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8_phase,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 23:
            if(m >= 17 || k >= 6144)
                return {m <= 16 ? GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3_phase :
                                  GroupedGemmInstanceId::bf16_nn_decode_coop_k32_d4_w8_phase,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 24:
            if(m >= 17 || k >= 6144)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k32_a2_b3_phase,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 25:
            return {GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2,
                    GroupedGemmSelectionReason::bf16_nn_decode_mainline};
        case 26:
            if(m >= 17)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 27:
            if(problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {m <= 16 ? GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2 :
                                  GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 28:
            if(problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_k64_d2,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 29:
            if(n % 256 == 0)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d2_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 30:
            if(n % 256 == 0 && problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d2_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 31:
            if(n % 256 == 0)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d3_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 32:
            if(n % 256 == 0 && problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_k32_d3_w8,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 33:
            if(n % 256 == 0)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d2,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 34:
            if(n % 256 == 0 && problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d2,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 35:
            if(n % 256 == 0)
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d3,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        case 36:
            if(n % 256 == 0 && problem.group_count <= 32 && (m >= 17 || k >= 6144))
                return {GroupedGemmInstanceId::bf16_nn_decode_coop_n256_b_reuse_d3,
                        GroupedGemmSelectionReason::bf16_nn_decode_mainline};
            break;
        default: break;
        }
    }
    if(policy.enable_bf16_nn_decode_short && decode_common && n >= 128 &&
       problem.group_count <= 32 && k >= 64 && k < 2048 && k % 32 == 0 &&
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, 64, 32) <= kGroupedGemmV7MaxTiles)
    {
        if(policy.bf16_nn_decode_short_variant == 0)
            return {GroupedGemmInstanceId::bf16_nn_decode_short_n64,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
        if(policy.bf16_nn_decode_short_variant == 1)
            return {GroupedGemmInstanceId::bf16_nn_decode_short_n32,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
        if(policy.bf16_nn_decode_short_variant == 2 && k % 64 == 0)
            return {GroupedGemmInstanceId::bf16_nn_decode_short_shared,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
        if(policy.bf16_nn_decode_short_variant == 3 && k % 64 == 0)
            return {GroupedGemmInstanceId::bf16_nn_decode_direct_short_xor,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
        if(policy.bf16_nn_decode_short_variant == 4 && k % 64 == 0)
            return {GroupedGemmInstanceId::bf16_nn_decode_direct_short_early,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
        if(policy.bf16_nn_decode_short_variant == 5 && k % 64 == 0)
            return {GroupedGemmInstanceId::bf16_nn_decode_direct_short_both,
                    GroupedGemmSelectionReason::bf16_nn_decode_short};
    }
    // Opt-in measured range. Device kargs retain the existing dense NN storage
    // contract: packed, aligned A/B/C; no host readback of their device pointers.
    if(policy.enable_bf16_nn_decode_depth8 && problem.arch == GroupedGemmArch::gfx936 &&
       is_bf16 && is_nn && problem.m.kind == GroupedGemmDimensionKind::common &&
       !has_variable_k && !problem.group_lengths_are_ragged &&
       !problem.may_have_empty_groups && problem.group_count == 64 &&
       m >= 1 && m <= 32 && n == 256 && k == 512 &&
       grouped_gemm_v7_experiment_tile_bound(problem, 16, 128) <= kGroupedGemmV7MaxTiles)
        return {GroupedGemmInstanceId::bf16_nn_decode_depth8,
                GroupedGemmSelectionReason::bf16_nn_decode_depth8};
    // Initial experimental contract: common dense BF16 NN, M<=64. The new
    // kernel has no relationship to the V7 policy or its tile/epilogue rules.
    if(policy.enable_bf16_nn_decode_wave && problem.arch == GroupedGemmArch::gfx936 &&
       is_bf16 && is_nn && problem.m.kind == GroupedGemmDimensionKind::common &&
       !has_variable_k && !problem.group_lengths_are_ragged &&
       !problem.may_have_empty_groups && problem.group_count <= 1024 &&
       m <= 64 && n % 128 == 0 && k >= 32 && k <= 4096 && k % 32 == 0 &&
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, 16, 128) <= kGroupedGemmV7MaxTiles)
        return {GroupedGemmInstanceId::bf16_nn_decode_wave,
                GroupedGemmSelectionReason::bf16_nn_decode_wave};
    const bool has_integral_device_m_capacity =
        problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity &&
        problem.m.extent % problem.group_count == 0;
    const bool logical_k_tail =
        is_bf16_tn && has_variable_k && policy.enable_bf16_tn_logical_k_tail;
    const std::int64_t large_tile_m_threshold =
        problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity ? 1920 : 2048;

    if(policy.enable_p3_fp16_tn_m1024_k4096_route && !is_bf16 && is_tn &&
       problem.m.kind == GroupedGemmDimensionKind::common && m == 1024 && n == 4096 &&
       has_variable_k && k % problem.group_count == 0 && effective_k == 4096 &&
       !problem.may_have_empty_groups && !problem.group_lengths_are_ragged)
    {
        if(problem.arch == GroupedGemmArch::gfx936)
        {
            return {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
                    GroupedGemmSelectionReason::p3_fp16_tn_m1024_k4096_route};
        }
        if(problem.arch == GroupedGemmArch::gfx938)
        {
            return {GroupedGemmInstanceId::gfx938_mls_large_256,
                    GroupedGemmSelectionReason::p3_fp16_tn_m1024_k4096_route};
        }
    }

    // v7 BF16 NN / TN instances (opt-in; see policy.enable_bf16_v7_v8*).
    //
    // The gates are the kernels' own contract, not a shape whitelist:
    //   K % 16 == 0 and K >= 64   -- the 16-deep k-stage and the 4-stage ring
    //   N % 256 == 0              -- the LDS-staged epilogue stores whole 16 B
    //                                row segments and carries no column
    //                                predicate, so a partial N tile would write
    //                                past the row
    //   common K                  -- the k-advance of both streams is per-launch
    //   non-empty groups          -- not validated for the walk yet; keep the
    //                                fallback until they are
    //   G <= kGroupedGemmV7NnMaxGroups (NN) / kGroupedGemmV8TnMaxGroups (TN)
    //                             -- the group counts each kernel's group
    //                                lookup is validated for
    //   tile bound <= kGroupedGemmV7MaxTiles
    //                             -- block ids and tile prefix sums are index_t
    //   gfx936                    -- the kernels' ISA contract (v7 loads, the
    //                                CACHE_SWIZZLE descriptor bit)
    // TN additionally requires an M that the launcher can address: its A panel
    // is [k, sum(M_g)] with a host-known pitch, and the panel load covers a whole
    // 256-wide column tile, so a ragged last tile reads up to 256 columns past
    // the group's last column. Reading into the NEXT group's columns is harmless
    // (those rows are masked on the store), but the read past the LAST group's
    // columns must land in storage the caller owns -- the reference kernel's
    // documented tail-pad obligation, which the selector cannot verify. Ragged M
    // is therefore allowed; the caller owns the pad.
    // Independent vector-aligned N tail. Neither the original no-column-
    // predicate epilogue nor its contract changes. B loads and C stores both
    // mask complete eight-BF16 vectors; reject smaller alignment/ragged inputs.
    const auto tail_tile_n = policy.bf16_v7_nn_tail_tile_n;
    if(policy.enable_bf16_v7_nn && policy.enable_bf16_v7_nn_tail128 &&
       (tail_tile_n == 128 || tail_tile_n == 256) &&
       problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
       problem.m.kind == GroupedGemmDimensionKind::common && !has_variable_k &&
       !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
       problem.group_count <= kGroupedGemmV7NnMaxGroups && m <= 128 &&
       k >= 64 && k <= 4096 && k % 16 == 0 && n >= 128 && n % 8 == 0 && n % 128 != 0 &&
       m <= std::int64_t{2147483647} / (2 * k) &&
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, 128, tail_tile_n) <= kGroupedGemmV7MaxTiles)
        return tail_tile_n == 256
            ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_tail256,
                                   GroupedGemmSelectionReason::v7_bf16_nn_tail256}
            : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_tail128,
                                   GroupedGemmSelectionReason::v7_bf16_nn_tail128};
    // Experimental small-M NN route. Whole N tiles only, so N=1232 stays
    // on its original fallback. Ragged/device M and empty groups fall back.
    const auto experiment = policy.bf16_v7_nn_small_m_experiment;
    const bool k32_experiment = experiment == 2 || experiment == 4 || experiment == 5;
    const auto experiment_m = experiment == 3 ? 128 : 64;
    const auto experiment_n = experiment == 4 ? 64 : 128;
    if(policy.enable_bf16_v7_nn && experiment >= 1 && experiment <= 7 &&
       (experiment < 5 || m <= 64) &&
       problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
       problem.m.kind == GroupedGemmDimensionKind::common && !has_variable_k &&
       !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
       problem.group_count <= kGroupedGemmV7NnMaxGroups && m <= 256 &&
       k >= (k32_experiment ? 128 : 64) && k <= 4096 &&
       k % (k32_experiment ? 32 : 16) == 0 && n % experiment_n == 0 &&
       // The D2L SRD/soffset path uses 32-bit byte ranges and signed arithmetic.
       // Limit the contiguous per-group B span before narrowing to those fields.
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, experiment_m, experiment_n) <=
           kGroupedGemmV7MaxTiles)
    {
        switch(experiment)
        {
        case 1: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_drain,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_drain};
        case 2: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k32};
        case 3: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_square128,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_square128};
        case 4: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k32n64};
        case 5: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32_steady,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k32_steady};
        case 6: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k16_steady,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k16_steady};
        case 7: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_noslc,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_noslc};
        }
    }
    const auto small_n = policy.bf16_v7_nn_small_m_tile_n;
    if(policy.enable_bf16_v7_nn && policy.enable_bf16_v7_nn_small_m &&
       problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
       problem.m.kind == GroupedGemmDimensionKind::common && !has_variable_k &&
       !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
       problem.group_count <= kGroupedGemmV7NnMaxGroups && m <= 256 &&
       k >= 64 && k <= 4096 && k % 16 == 0 &&
       (small_n == 64 || small_n == 128) && n % small_n == 0 &&
       grouped_gemm_v7_small_m_tile_bound(problem, small_n) <= kGroupedGemmV7MaxTiles)
    {
        return small_n == 128
            ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_small_m128,
                                   GroupedGemmSelectionReason::v7_bf16_nn_small_m128}
            : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_small_m64,
                                   GroupedGemmSelectionReason::v7_bf16_nn_small_m64};
    }
    const bool v7_contract =
        problem.arch == GroupedGemmArch::gfx936 && !has_variable_k &&
        grouped_gemm_v7_tile_bound(problem) <= kGroupedGemmV7MaxTiles &&
        !problem.may_have_empty_groups && k >= 64 && k % 16 == 0 && n % 256 == 0;
    if(v7_contract && problem.group_count <= kGroupedGemmV7NnMaxGroups && is_bf16 &&
       is_nn && policy.enable_bf16_v7_nn)
    {
        // P6f factor isolation. An explicit opt-in controls only a valid common
        // NN V7 contract; this is not an ID/shape dispatch override.
        if((policy.bf16_v7_nn_cache_walk == 1 || policy.bf16_v7_nn_cache_walk == 5 ||
            policy.bf16_v7_nn_cache_walk == 8) &&
           problem.m.kind == GroupedGemmDimensionKind::common &&
           !problem.group_lengths_are_ragged && k <= 1073741823 &&
           m <= std::int64_t{2147483647} / (2 * k) &&
           n <= std::int64_t{2147483647} / (2 * k))
        {
            const bool no_slc = policy.enable_bf16_v7_nn_a_noslc;
            if(policy.bf16_v7_nn_cache_walk == 5)
                return no_slc
                    ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,
                                           GroupedGemmSelectionReason::v7_bf16_nn_cache_noslc_m01_5}
                    : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_cache_m01_5,
                                           GroupedGemmSelectionReason::v7_bf16_nn_cache_m01_5};
            if(policy.bf16_v7_nn_cache_walk == 8)
                return no_slc
                    ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_a_noslc_m01_8,
                                           GroupedGemmSelectionReason::v7_bf16_nn_a_noslc_m01_8}
                    : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_m01_8,
                                           GroupedGemmSelectionReason::v7_bf16_nn_m01_8};
            return no_slc
                ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_a_noslc,
                                       GroupedGemmSelectionReason::v7_bf16_nn_a_noslc}
                : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn,
                                       GroupedGemmSelectionReason::v7_bf16_nn};
        }
        // P6f opt-in bounded cache policy, validated as a complete attribute
        // domain. No caller tile knowledge or per-shape lookup is required.
        if(policy.enable_bf16_v7_nn_cache_grid &&
           problem.m.kind == GroupedGemmDimensionKind::common &&
           !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
           m > 1024 && m <= 2048 && n > 4096 && k >= 3072 && k <= 4096 &&
           m <= std::int64_t{2147483647} / (2 * k) &&
           n <= std::int64_t{2147483647} / (2 * k))
            return {GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,
                    GroupedGemmSelectionReason::v7_bf16_nn_cache_noslc_m01_5};
        // Experimental cache-only variant: keep the original tile walk.
        if(policy.enable_bf16_v7_nn_a_noslc &&
           problem.m.kind == GroupedGemmDimensionKind::common &&
           !problem.group_lengths_are_ragged &&
           // Preserve representable per-group A/B ranges in this new instance.
           k <= 1073741823 &&
           m <= std::int64_t{2147483647} / (2 * k) &&
           n <= std::int64_t{2147483647} / (2 * k))
        {
            if(policy.enable_bf16_v7_nn_m01_8 && k > kGroupedGemmV7NnM01_8KThreshold)
                return {GroupedGemmInstanceId::v7_bf16_nn_a_noslc_m01_8,
                        GroupedGemmSelectionReason::v7_bf16_nn_a_noslc_m01_8};
            return {GroupedGemmInstanceId::v7_bf16_nn_a_noslc,
                    GroupedGemmSelectionReason::v7_bf16_nn_a_noslc};
        }
        // K is common here (v7_contract), so k is every group's K.
        if(policy.enable_bf16_v7_nn_m01_8 && k > kGroupedGemmV7NnM01_8KThreshold)
        {
            return {GroupedGemmInstanceId::v7_bf16_nn_m01_8,
                    GroupedGemmSelectionReason::v7_bf16_nn_m01_8};
        }
        return {GroupedGemmInstanceId::v7_bf16_nn,
                GroupedGemmSelectionReason::v7_bf16_nn};
    }
    if(v7_contract && problem.group_count <= kGroupedGemmV8TnMaxGroups && is_bf16 &&
       is_tn && policy.enable_bf16_v8_tn)
    {
        return {GroupedGemmInstanceId::v8_bf16_tn,
                GroupedGemmSelectionReason::v8_bf16_tn};
    }

    const bool bw_arch_layout =
        problem.arch == GroupedGemmArch::gfx936 ||
        (problem.arch == GroupedGemmArch::gfx938 && is_bf16_tn);
    const bool selected_layout = is_nt || is_nn || is_bf16_tn;
    if(policy.enable_bw_family && bw_arch_layout && selected_layout &&
       effective_m >= large_tile_m_threshold && n >= 2048 && k >= 2048 && n % 256 == 0 &&
       (k % 64 == 0 || logical_k_tail))
    {
        if(problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
           policy.enable_gfx936_bf16_nn_promoted && problem.group_count == 16 &&
           problem.m.kind == GroupedGemmDimensionKind::device_lengths_with_capacity &&
           effective_m >= 1920 && effective_m <= 4096)
        {
            if(n == 7168 && k == 4096)
            {
                return {GroupedGemmInstanceId::bw_family_bf16_nn_fixed_srd_lds8,
                        GroupedGemmSelectionReason::bw_family};
            }
            if(n == 2048 && k == 7168)
            {
                return {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
                        GroupedGemmSelectionReason::bw_family};
            }
        }

        if(policy.enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 &&
           problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
           problem.group_count == 8 &&
           (problem.m.kind == GroupedGemmDimensionKind::common ||
            has_integral_device_m_capacity) &&
           effective_m >= 2048 && effective_m <= 2221 && n == 2048 && k == 7168)
        {
            // ID1002 consumes the device-built per-group descriptors, as does the
            // existing G16 route above. Primus reports the G8 M summary as total
            // packed capacity, so use its integral per-group extent here; the
            // ragged/empty flags remain valid and do not imply a common length.
            return {GroupedGemmInstanceId::bw_family_bf16_nn_fast_vmem3,
                    GroupedGemmSelectionReason::p2_bf16_g8_gfx936_fc2_nn_fast_vmem3};
        }

        if(is_bf16_tn)
        {
            return {logical_k_tail ? GroupedGemmInstanceId::bw_family_logical_k_tail
                                   : GroupedGemmInstanceId::bw_family_blas_transposed,
                    GroupedGemmSelectionReason::bw_family};
        }
        if(policy.enable_bw_selected)
        {
            return {GroupedGemmInstanceId::bw_family_selected,
                    GroupedGemmSelectionReason::bw_family};
        }
    }

    if(policy.enable_gfx936_v3 && problem.arch == GroupedGemmArch::gfx936 &&
       effective_m >= large_tile_m_threshold && n >= 2048 && k >= 2048 && n % 256 == 0 &&
       k % 64 == 0)
    {
        if(is_nt)
        {
            return {GroupedGemmInstanceId::gfx936_v3_256_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
        if((is_nn || is_tn) && policy.enable_gfx936_dsreadm)
        {
            return {GroupedGemmInstanceId::gfx936_v3_dsreadm_backward_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
        if(is_nn)
        {
            return {GroupedGemmInstanceId::gfx936_v3_default_backward_m_only_padding,
                    GroupedGemmSelectionReason::gfx936_v3};
        }
    }

    if(policy.enable_p2_bf16_g8_gfx938_v5_fc1_route && is_bf16 && is_nt &&
       problem.arch == GroupedGemmArch::gfx938 && problem.group_count == 8 &&
       problem.m.kind == GroupedGemmDimensionKind::common && m >= 1918 && m <= 1921 &&
       n == 4096 && k == 7168)
    {
        return {m % 128 == 0 ? GroupedGemmInstanceId::v4_128_m_only_padding
                            : GroupedGemmInstanceId::v4_64_nonpadding,
                GroupedGemmSelectionReason::p2_bf16_g8_gfx938_v5_fc1_route};
    }

    if(policy.enable_gfx938_mls && problem.arch == GroupedGemmArch::gfx938 &&
       effective_m >= 512 && n >= 512 && k >= 512)
    {
        if(effective_m >= large_tile_m_threshold && n >= 2048 &&
           policy.enable_gfx938_mls_large)
        {
            return {GroupedGemmInstanceId::gfx938_mls_large_256,
                    GroupedGemmSelectionReason::gfx938_mls};
        }
        if(is_nt)
        {
            return {GroupedGemmInstanceId::gfx938_mls_small_128,
                    GroupedGemmSelectionReason::gfx938_mls};
        }
    }

    if(effective_m >= 512 && n >= 512 && k >= 512 && m % 128 == 0 && n % 128 == 0 &&
       k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v4_128_m_only_padding,
                GroupedGemmSelectionReason::v4_aligned};
    }
    if(is_bf16 && m >= 512 && n >= 512 && k >= 512 && m % 128 == 0 && n % 128 == 0)
    {
        return {GroupedGemmInstanceId::v4_128_full_padding,
                GroupedGemmSelectionReason::v4_bf16_padding};
    }
    if(policy.enable_p2_bf16_g8_v4_64_m01_4_gfx936 && is_bf16 &&
       problem.group_count == 8 && problem.m.kind == GroupedGemmDimensionKind::common &&
       m >= 1918 && m <= 1921 && m % 128 != 0 && n >= 2048 && k >= 2048 &&
       n % 128 == 0 && k % 64 == 0 && problem.arch == GroupedGemmArch::gfx936 &&
       (is_nt || is_nn))
    {
        return {GroupedGemmInstanceId::v4_64_m01_4_gfx936,
                GroupedGemmSelectionReason::p2_bf16_g8_v4_64_m01_4_gfx936};
    }

    // Earlier P1 V4 M4/K32/M-threshold/128x64/grid2 experiments did not beat
    // these architecture-local V3 single-LDS routes and were retired.
    if(policy.enable_p1_family_a_v3_128_gfx938 &&
       problem.arch == GroupedGemmArch::gfx938 && !is_bf16 && is_nn && effective_m >= 33 &&
       n >= 1024 && k >= 1024 && n % 128 != 0 && k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v3_128_m4_full_padding,
                GroupedGemmSelectionReason::p1_family_a_v3_128_gfx938};
    }
    if(policy.enable_p1_family_a_v3_128_gfx936 &&
       problem.arch == GroupedGemmArch::gfx936 && !is_bf16 && is_nn && effective_m >= 33 &&
       n >= 1024 && k >= 1024 && n % 128 != 0 && k % 64 == 0)
    {
        return {GroupedGemmInstanceId::v3_128_m4_full_padding_gfx936,
                GroupedGemmSelectionReason::p1_family_a_v3_128_gfx936};
    }
    // The wider M<=10 Family-B experiment had regressions; retain the proven M<=8 gate.
    if(policy.enable_p1_family_b_v6_m32_mle8_gfx936 &&
       problem.arch == GroupedGemmArch::gfx936 && !is_bf16 && is_nn &&
       effective_m <= 8 && n == 3072 && k == 1232)
    {
        return {GroupedGemmInstanceId::v6_m32_nonpadding_mle8_family_b_gfx936,
                GroupedGemmSelectionReason::p1_family_b_v6_m32_mle8_gfx936};
    }
    if(n % 128 == 0)
    {
        return {GroupedGemmInstanceId::v4_64_nonpadding,
                GroupedGemmSelectionReason::v4_n_aligned};
    }
    return {GroupedGemmInstanceId::v4_64_full_padding,
            GroupedGemmSelectionReason::v4_fallback};
}

constexpr std::string_view grouped_gemm_arch_name(GroupedGemmArch arch)
{
    return arch == GroupedGemmArch::gfx936 ? "gfx936"
           : arch == GroupedGemmArch::gfx938 ? "gfx938"
           : arch == GroupedGemmArch::gfx946 ? "gfx946"
                                              : "unsupported";
}

constexpr std::string_view grouped_gemm_data_type_name(GroupedGemmDataType data_type)
{
    return data_type == GroupedGemmDataType::fp16 ? "fp16"
           : data_type == GroupedGemmDataType::bf16 ? "bf16"
                                                     : "unsupported";
}

constexpr std::string_view grouped_gemm_layout_name(GroupedGemmLayout layout)
{
    return layout == GroupedGemmLayout::nt ? "NT"
           : layout == GroupedGemmLayout::nn ? "NN"
           : layout == GroupedGemmLayout::tn ? "TN"
                                              : "unsupported";
}

inline std::string stable_grouped_gemm_instance_id(const GroupedGemmProblem& problem,
                                                   const GroupedGemmSelection& selection)
{
    const auto* candidate = find_grouped_gemm_candidate(selection.instance_id);
    if(candidate == nullptr)
    {
        return {};
    }
    std::string id{"ck.gg."};
    id.append(grouped_gemm_arch_name(problem.arch));
    id.push_back('.');
    id.append(grouped_gemm_data_type_name(problem.data_type));
    id.push_back('.');
    id.append(grouped_gemm_layout_name(problem.layout));
    id.push_back('.');
    id.append(candidate->canonical_suffix);
    return id;
}

} // namespace ck_tile
