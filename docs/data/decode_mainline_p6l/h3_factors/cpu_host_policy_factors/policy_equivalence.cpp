#include <cstdlib>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_DECODE_BF16 1
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED 1
#define CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16 1
inline bool ck_tile_hcu_grouped_gemm_bf16_nn_fixed_srd_lds8_enabled()
{
    const char* disable =
        std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_NN_FIXED_SRD_LDS8");
    return !(disable != nullptr && disable[0] == '1' && disable[1] == '\0');
}

// CK-owned gfx936 BF16 TN logical-K-tail default. The candidate has an
// explicit rollback switch so downstream integrations can perform same-
// process A/B without changing the aligned/generic fallback portfolio.
inline bool ck_tile_hcu_grouped_gemm_bf16_tn_logical_k_tail_enabled()
{
    const char* disable = std::getenv(
        "CK_TILE_GROUPED_GEMM_DISABLE_GFX936_BF16_TN_LOGICAL_K_TAIL");
    return !(disable != nullptr && disable[0] == '1' && disable[1] == '\0');
}

// v7 BF16 NN / TN instances. These routes are OPT-IN (the selector policy
// fields default to false, see grouped_gemm_selector.hpp), so unlike the two
// switches above the enable side is the one that has to be set explicitly. The
// disable side still wins, which is what makes a same-process A/B able to pin
// either state:
//   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN=1  -> route BF16 NN to instance 5000
//                                               (5002 for K > 4096)
//   CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN=1  -> route BF16 TN to instance 5001
//   CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_V8*=1  -> force the route off
inline bool ck_tile_hcu_grouped_gemm_bf16_v7_nn_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0')
        return false;
    const char* enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

// Inside the v7 NN route, the large-K tile walk (5002) is on by default:
//   CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8=1 -> keep every v7 NN problem on 5000
inline bool ck_tile_hcu_grouped_gemm_bf16_v7_nn_m01_8_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_M01_8");
    return !(disable != nullptr && disable[0] == '1' && disable[1] == '\0');
}

// Experimental value 64/128 selects the logical N tile; disable wins.
inline int ck_tile_hcu_grouped_gemm_bf16_v7_nn_small_m_tile_n()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0')
        return 0;
    const char* value = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M");
    if(value == nullptr)
        return 0;
    if(value[0] == '6' && value[1] == '4' && value[2] == '\0')
        return 64;
    if(value[0] == '1' && value[1] == '2' && value[2] == '8' && value[3] == '\0')
        return 128;
    return 0;
}

// Experimental A-load cache policy, default off; disable wins.
inline bool ck_tile_hcu_grouped_gemm_bf16_v7_nn_a_noslc_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_A_NOSLC");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0') return false;
    const char* enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_A_NOSLC");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

inline bool ck_tile_hcu_grouped_gemm_bf16_v8_tn_enabled()
{
    const char* disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V8_TN");
    if(disable != nullptr && disable[0] == '1' && disable[1] == '\0')
        return false;
    const char* enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V8_TN");
    return enable != nullptr && enable[0] == '1' && enable[1] == '\0';
}

__attribute__((noinline)) int read_policy_0(const ck_tile::GroupedGemmProblem& problem, ck_tile::GroupedGemmSelectorPolicy& policy, unsigned disabled_flags) {
ck_tile_hcu_grouped_gemm_problem_v1 record{}; record.policy_disable_flags=disabled_flags; const auto* source=&record;
    const auto disabled = source->policy_disable_flags;
    policy.enable_bw_family =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1) == 0;
    policy.enable_gfx936_v3 =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1) == 0;
    policy.enable_gfx936_dsreadm =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1) == 0;
    policy.enable_gfx938_mls =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1) == 0;
    policy.enable_gfx936_bf16_nn_promoted =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_BF16_NN_PROMOTED_V1) == 0;
    policy.enable_bf16_tn_logical_k_tail =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BF16_TN_LOGICAL_K_TAIL_V1) == 0;

    // CK-owned rollback switches. The selector itself is a pure function of
    // Problem+Policy; env handling stays in this adapter so same-process A/B
    // does not require the consumer to translate those names into flags.
    if(!ck_tile_hcu_grouped_gemm_bf16_nn_fixed_srd_lds8_enabled())
        policy.enable_gfx936_bf16_nn_promoted = false;
    if(!ck_tile_hcu_grouped_gemm_bf16_tn_logical_k_tail_enabled())
        policy.enable_bf16_tn_logical_k_tail = false;

    // v7 NN / TN: opt-in, so the env switch is the ENABLE side here (the
    // selector's own default is false). See
    // grouped_gemm_bw_family_selected.hpp for the switch semantics.
    policy.enable_bf16_v7_nn = ck_tile_hcu_grouped_gemm_bf16_v7_nn_enabled();
    policy.enable_bf16_v8_tn = ck_tile_hcu_grouped_gemm_bf16_v8_tn_enabled();
    policy.enable_bf16_v7_nn_m01_8 = ck_tile_hcu_grouped_gemm_bf16_v7_nn_m01_8_enabled();
    const int small_n = ck_tile_hcu_grouped_gemm_bf16_v7_nn_small_m_tile_n();
    policy.enable_bf16_v7_nn_small_m = small_n != 0;
    policy.bf16_v7_nn_small_m_tile_n = small_n;
    policy.enable_bf16_v7_nn_a_noslc = ck_tile_hcu_grouped_gemm_bf16_v7_nn_a_noslc_enabled();
    const char* cache_walk = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK");
    const char* cache_walk_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_WALK");
    if(cache_walk != nullptr && cache_walk[0] != '\0' && cache_walk[1] == '\0' &&
       (cache_walk[0] == '1' || cache_walk[0] == '5' || cache_walk[0] == '8') &&
       !(cache_walk_disable != nullptr && cache_walk_disable[0] == '1' && cache_walk_disable[1] == '\0'))
        policy.bf16_v7_nn_cache_walk = cache_walk[0] - '0';
    const char* tail_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL");
    const char* tail_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_N_TAIL");
    policy.enable_bf16_v7_nn_tail128 = tail_enable != nullptr && (tail_enable[0] == '1' || tail_enable[0] == '2') && tail_enable[1] == '\0' &&
        !(tail_disable != nullptr && tail_disable[0] == '1' && tail_disable[1] == '\0');
    policy.bf16_v7_nn_tail_tile_n = tail_enable != nullptr && tail_enable[0] == '2' ? 256 : 128;
    const char* cache_grid = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID");
    const char* cache_grid_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_GRID");
    policy.enable_bf16_v7_nn_cache_grid = cache_grid != nullptr && cache_grid[0] == '1' && cache_grid[1] == '\0' &&
        !(cache_grid_disable != nullptr && cache_grid_disable[0] == '1' && cache_grid_disable[1] == '\0');
    const char* depth8_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_DEPTH8");
    const char* depth8_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_DEPTH8");
    policy.enable_bf16_nn_decode_depth8 = depth8_enable != nullptr &&
        depth8_enable[0] == '1' && depth8_enable[1] == '\0' &&
        !(depth8_disable != nullptr && depth8_disable[0] == '1' && depth8_disable[1] == '\0');
    const auto decode_switch = [](const char* enable, const char* disable) {
        const char* e = std::getenv(enable);
        const char* d = std::getenv(disable);
        return e != nullptr && e[0] == '1' && e[1] == '\0' &&
            !(d != nullptr && d[0] == '1' && d[1] == '\0');
    };
    const auto decode_variant = [](const char* name, int maximum) {
        const char* v = std::getenv(name);
        if(v == nullptr) return 0;
        if(v[0] < '0' || v[0] > '9') return -1;
        int value = v[0] - '0';
        if(v[1] != '\0')
        {
            if(value == 0 || v[1] < '0' || v[1] > '9' || v[2] != '\0') return -1;
            value = value * 10 + v[1] - '0';
        }
        return value <= maximum ? value : -1;
    };
    policy.enable_bf16_nn_decode_mainline = decode_switch(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_MAINLINE",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_MAINLINE");
    policy.bf16_nn_decode_mainline_variant = decode_variant(
        "CK_TILE_GROUPED_GEMM_BF16_NN_DECODE_MAINLINE_VARIANT", 11);
    policy.enable_bf16_nn_decode_short = decode_switch(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_SHORT",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_SHORT");
    policy.bf16_nn_decode_short_variant = decode_variant(
        "CK_TILE_GROUPED_GEMM_BF16_NN_DECODE_SHORT_VARIANT", 5);
    const char* decode_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_WAVE");
    const char* decode_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_WAVE");
    policy.enable_bf16_nn_decode_wave = decode_enable != nullptr &&
        decode_enable[0] == '1' && decode_enable[1] == '\0' &&
        !(decode_disable != nullptr && decode_disable[0] == '1' && decode_disable[1] == '\0');
    const char* bvector_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_BVECTOR");
    const char* bvector_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_BVECTOR");
    if(bvector_enable != nullptr && (bvector_enable[0] == '1' || bvector_enable[0] == '2') &&
       bvector_enable[1] == '\0' &&
       !(bvector_disable != nullptr && bvector_disable[0] == '1' && bvector_disable[1] == '\0'))
        policy.bf16_nn_decode_bvector = bvector_enable[0] - '0';
#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_DECODE_BF16)
    policy.enable_bf16_nn_decode_depth8 = false;
    policy.enable_bf16_nn_decode_mainline = false;
    policy.enable_bf16_nn_decode_short = false;
    policy.enable_bf16_nn_decode_wave = false;
    policy.bf16_nn_decode_bvector = 0;
#endif
    // Local experimental adapter: no public ABI flags or shared provider header.
    const char* experiment = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT");
    const char* experiment_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT");
    const char* small_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M");
    const auto experiment_is_disabled = [](const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; };
    if(experiment != nullptr && experiment[0] >= '1' && experiment[0] <= '7' &&
       experiment[1] == '\0' && !experiment_is_disabled(experiment_disable) && !experiment_is_disabled(small_disable))
        policy.bf16_v7_nn_small_m_experiment = experiment[0] - '0';


#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
    // Do not name InstanceId 1001/1002 when this TU cannot dispatch them.
    policy.enable_gfx936_bf16_nn_promoted = false;
#endif

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16)
    // Same rule for InstanceId 5000/5001/5002: this archive does not carry the
    // v7 entries (gfx936-only, dedicated codegen contract), so the route must
    // not be selectable.
    policy.enable_bf16_v7_nn = false;
    policy.enable_bf16_v8_tn = false;
    policy.enable_bf16_v7_nn_m01_8 = false;
    policy.enable_bf16_v7_nn_small_m = false;
    policy.enable_bf16_v7_nn_a_noslc = false;
    policy.bf16_v7_nn_small_m_experiment = 0;
    policy.bf16_v7_nn_cache_walk = 0;
    policy.enable_bf16_v7_nn_cache_grid = false;
    policy.enable_bf16_v7_nn_tail128 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx938 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE)
    policy.enable_p1_family_b_v6_m32_mle8_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE)
    policy.enable_p2_bf16_g8_v4_64_m01_4_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE)
    policy.enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX938_V5_FC1_ROUTE_DISABLE)
    policy.enable_p2_bf16_g8_gfx938_v5_fc1_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)
    policy.enable_p3_fp16_tn_m1024_k4096_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
    // These architecture-specialized code objects are retained for non-RDC
    // builds, but real gfx936/gfx938 validation shows they are not correct
    // after device linking. Keep the selected InstanceId honest by applying
    // a build-capability policy before selection instead of dispatching a
    // different kernel behind the specialized ID.
    policy.enable_bw_selected      = false;
    policy.enable_gfx938_mls_large = false;
#endif

    return ck_tile::is_valid_grouped_gemm_problem(problem)
               ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
               : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}
__attribute__((noinline)) int read_policy_1(const ck_tile::GroupedGemmProblem& problem, ck_tile::GroupedGemmSelectorPolicy& policy, unsigned disabled_flags) {
ck_tile_hcu_grouped_gemm_problem_v1 record{}; record.policy_disable_flags=disabled_flags; const auto* source=&record;
    // Only explicit new decode requests can skip unrelated policy parsing.
    // Read these existing switches once and keep every legacy fallback below.
    const auto decode_switch = [](const char* enable, const char* disable) {
        const char* e = std::getenv(enable);
        const char* d = std::getenv(disable);
        return e != nullptr && e[0] == '1' && e[1] == '\0' &&
            !(d != nullptr && d[0] == '1' && d[1] == '\0');
    };
    const auto decode_variant = [](const char* name, int maximum) {
        const char* v = std::getenv(name);
        if(v == nullptr) return 0;
        if(v[0] < '0' || v[0] > '9') return -1;
        int value = v[0] - '0';
        if(v[1] != '\0')
        {
            if(value == 0 || v[1] < '0' || v[1] > '9' || v[2] != '\0') return -1;
            value = value * 10 + v[1] - '0';
        }
        return value <= maximum ? value : -1;
    };
    policy.enable_bf16_nn_decode_mainline = decode_switch(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_MAINLINE",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_MAINLINE");
    policy.bf16_nn_decode_mainline_variant = decode_variant(
        "CK_TILE_GROUPED_GEMM_BF16_NN_DECODE_MAINLINE_VARIANT", 16);
    policy.enable_bf16_nn_decode_short = decode_switch(
        "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_SHORT",
        "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_SHORT");
    policy.bf16_nn_decode_short_variant = decode_variant(
        "CK_TILE_GROUPED_GEMM_BF16_NN_DECODE_SHORT_VARIANT", 5);
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_DECODE_BF16)
    const bool priority_decode_requested =
        (policy.enable_bf16_nn_decode_mainline &&
         policy.bf16_nn_decode_mainline_variant >= 6 &&
         policy.bf16_nn_decode_mainline_variant <= 16) ||
        (policy.enable_bf16_nn_decode_short &&
         policy.bf16_nn_decode_short_variant >= 3 &&
         policy.bf16_nn_decode_short_variant <= 5);
    if(priority_decode_requested && decode_switch(
           "CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_FAST_POLICY",
           "CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_FAST_POLICY"))
    {
        const auto priority_decode_candidate =
            ck_tile::select_grouped_gemm_candidate(problem, policy);
        const int priority_decode_id = static_cast<int>(priority_decode_candidate.instance_id);
        if((priority_decode_id >= 6013 && priority_decode_id <= 6018) ||
           (priority_decode_id >= 6020 && priority_decode_id <= 6023))
            return CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1;
    }
#endif

    const auto disabled = source->policy_disable_flags;
    policy.enable_bw_family =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BW_FAMILY_V1) == 0;
    policy.enable_gfx936_v3 =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_V3_V1) == 0;
    policy.enable_gfx936_dsreadm =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_DSREADM_V1) == 0;
    policy.enable_gfx938_mls =
        (disabled & CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX938_MLS_V1) == 0;
    policy.enable_gfx936_bf16_nn_promoted =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_GFX936_BF16_NN_PROMOTED_V1) == 0;
    policy.enable_bf16_tn_logical_k_tail =
        (disabled &
         CK_TILE_HCU_GROUPED_GEMM_POLICY_DISABLE_BF16_TN_LOGICAL_K_TAIL_V1) == 0;

    // CK-owned rollback switches. The selector itself is a pure function of
    // Problem+Policy; env handling stays in this adapter so same-process A/B
    // does not require the consumer to translate those names into flags.
    if(!ck_tile_hcu_grouped_gemm_bf16_nn_fixed_srd_lds8_enabled())
        policy.enable_gfx936_bf16_nn_promoted = false;
    if(!ck_tile_hcu_grouped_gemm_bf16_tn_logical_k_tail_enabled())
        policy.enable_bf16_tn_logical_k_tail = false;

    // v7 NN / TN: opt-in, so the env switch is the ENABLE side here (the
    // selector's own default is false). See
    // grouped_gemm_bw_family_selected.hpp for the switch semantics.
    policy.enable_bf16_v7_nn = ck_tile_hcu_grouped_gemm_bf16_v7_nn_enabled();
    policy.enable_bf16_v8_tn = ck_tile_hcu_grouped_gemm_bf16_v8_tn_enabled();
    policy.enable_bf16_v7_nn_m01_8 = ck_tile_hcu_grouped_gemm_bf16_v7_nn_m01_8_enabled();
    const int small_n = ck_tile_hcu_grouped_gemm_bf16_v7_nn_small_m_tile_n();
    policy.enable_bf16_v7_nn_small_m = small_n != 0;
    policy.bf16_v7_nn_small_m_tile_n = small_n;
    policy.enable_bf16_v7_nn_a_noslc = ck_tile_hcu_grouped_gemm_bf16_v7_nn_a_noslc_enabled();
    const char* cache_walk = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK");
    const char* cache_walk_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_WALK");
    if(cache_walk != nullptr && cache_walk[0] != '\0' && cache_walk[1] == '\0' &&
       (cache_walk[0] == '1' || cache_walk[0] == '5' || cache_walk[0] == '8') &&
       !(cache_walk_disable != nullptr && cache_walk_disable[0] == '1' && cache_walk_disable[1] == '\0'))
        policy.bf16_v7_nn_cache_walk = cache_walk[0] - '0';
    const char* tail_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL");
    const char* tail_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_N_TAIL");
    policy.enable_bf16_v7_nn_tail128 = tail_enable != nullptr && (tail_enable[0] == '1' || tail_enable[0] == '2') && tail_enable[1] == '\0' &&
        !(tail_disable != nullptr && tail_disable[0] == '1' && tail_disable[1] == '\0');
    policy.bf16_v7_nn_tail_tile_n = tail_enable != nullptr && tail_enable[0] == '2' ? 256 : 128;
    const char* cache_grid = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID");
    const char* cache_grid_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_GRID");
    policy.enable_bf16_v7_nn_cache_grid = cache_grid != nullptr && cache_grid[0] == '1' && cache_grid[1] == '\0' &&
        !(cache_grid_disable != nullptr && cache_grid_disable[0] == '1' && cache_grid_disable[1] == '\0');
    const char* depth8_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_DEPTH8");
    const char* depth8_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_DEPTH8");
    policy.enable_bf16_nn_decode_depth8 = depth8_enable != nullptr &&
        depth8_enable[0] == '1' && depth8_enable[1] == '\0' &&
        !(depth8_disable != nullptr && depth8_disable[0] == '1' && depth8_disable[1] == '\0');
    const char* decode_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_WAVE");
    const char* decode_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_WAVE");
    policy.enable_bf16_nn_decode_wave = decode_enable != nullptr &&
        decode_enable[0] == '1' && decode_enable[1] == '\0' &&
        !(decode_disable != nullptr && decode_disable[0] == '1' && decode_disable[1] == '\0');
    const char* bvector_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_NN_DECODE_BVECTOR");
    const char* bvector_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_NN_DECODE_BVECTOR");
    if(bvector_enable != nullptr && (bvector_enable[0] == '1' || bvector_enable[0] == '2') &&
       bvector_enable[1] == '\0' &&
       !(bvector_disable != nullptr && bvector_disable[0] == '1' && bvector_disable[1] == '\0'))
        policy.bf16_nn_decode_bvector = bvector_enable[0] - '0';
#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_DECODE_BF16)
    policy.enable_bf16_nn_decode_depth8 = false;
    policy.enable_bf16_nn_decode_mainline = false;
    policy.enable_bf16_nn_decode_short = false;
    policy.enable_bf16_nn_decode_wave = false;
    policy.bf16_nn_decode_bvector = 0;
#endif
    // Local experimental adapter: no public ABI flags or shared provider header.
    const char* experiment = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_SMALL_M_EXPERIMENT");
    const char* experiment_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M_EXPERIMENT");
    const char* small_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_SMALL_M");
    const auto experiment_is_disabled = [](const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; };
    if(experiment != nullptr && experiment[0] >= '1' && experiment[0] <= '7' &&
       experiment[1] == '\0' && !experiment_is_disabled(experiment_disable) && !experiment_is_disabled(small_disable))
        policy.bf16_v7_nn_small_m_experiment = experiment[0] - '0';


#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_GFX936_PROMOTED)
    // Do not name InstanceId 1001/1002 when this TU cannot dispatch them.
    policy.enable_gfx936_bf16_nn_promoted = false;
#endif

#if !defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16)
    // Same rule for InstanceId 5000/5001/5002: this archive does not carry the
    // v7 entries (gfx936-only, dedicated codegen contract), so the route must
    // not be selectable.
    policy.enable_bf16_v7_nn = false;
    policy.enable_bf16_v8_tn = false;
    policy.enable_bf16_v7_nn_m01_8 = false;
    policy.enable_bf16_v7_nn_small_m = false;
    policy.enable_bf16_v7_nn_a_noslc = false;
    policy.bf16_v7_nn_small_m_experiment = 0;
    policy.bf16_v7_nn_cache_walk = 0;
    policy.enable_bf16_v7_nn_cache_grid = false;
    policy.enable_bf16_v7_nn_tail128 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX938_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx938 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_A_V3_128_GFX936_DISABLE)
    policy.enable_p1_family_a_v3_128_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P1_FAMILY_B_V6_M32_MLE8_GFX936_DISABLE)
    policy.enable_p1_family_b_v6_m32_mle8_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_V4_64_M01_4_GFX936_DISABLE)
    policy.enable_p2_bf16_g8_v4_64_m01_4_gfx936 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX936_FC2_NN_FAST_VMEM3_DISABLE)
    policy.enable_p2_bf16_g8_gfx936_fc2_nn_fast_vmem3 = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P2_BF16_G8_GFX938_V5_FC1_ROUTE_DISABLE)
    policy.enable_p2_bf16_g8_gfx938_v5_fc1_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_P3_FP16_TN_M1024_K4096_ROUTE_DISABLE)
    policy.enable_p3_fp16_tn_m1024_k4096_route = false;
#endif

#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_RDC_BUILD)
    // These architecture-specialized code objects are retained for non-RDC
    // builds, but real gfx936/gfx938 validation shows they are not correct
    // after device linking. Keep the selected InstanceId honest by applying
    // a build-capability policy before selection instead of dispatching a
    // different kernel behind the specialized ID.
    policy.enable_bw_selected      = false;
    policy.enable_gfx938_mls_large = false;
#endif

    return ck_tile::is_valid_grouped_gemm_problem(problem)
               ? CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1
               : CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_UNSUPPORTED_V1;
}
int main() {
 std::string line; std::vector<std::string> prior;
 while(std::getline(std::cin,line)) {
  for(const auto& key:prior) unsetenv(key.c_str()); prior.clear();
  std::istringstream input(line); int id,arch,type,layout,groups,kind,empty,ragged,mode;
  long long m,n,k; unsigned disabled; size_t count;
  input>>id>>arch>>type>>layout>>m>>n>>k>>groups>>kind>>empty>>ragged>>disabled>>mode>>count;
  if(!input) return 2;
  for(size_t i=0;i<count;++i) {std::string key,value;input>>key>>value; if(value=="~")value=""; setenv(key.c_str(),value.c_str(),1);prior.push_back(key);}
  ck_tile::GroupedGemmProblem p{};p.arch=static_cast<ck_tile::GroupedGemmArch>(arch);
  p.data_type=static_cast<ck_tile::GroupedGemmDataType>(type);p.layout=static_cast<ck_tile::GroupedGemmLayout>(layout);
  p.group_count=groups;p.m=ck_tile::GroupedGemmDimensionSummary::Common(m);p.n=ck_tile::GroupedGemmDimensionSummary::Common(n);p.k=ck_tile::GroupedGemmDimensionSummary::Common(k);
  p.m.kind=static_cast<ck_tile::GroupedGemmDimensionKind>(kind);p.may_have_empty_groups=empty;p.group_lengths_are_ragged=ragged;
  ck_tile::GroupedGemmSelectorPolicy a{},b{};
  const int ra=read_policy_0(p,a,disabled),rb=read_policy_1(p,b,disabled);
  const auto sa=ck_tile::select_grouped_gemm_candidate(p,a),sb=ck_tile::select_grouped_gemm_candidate(p,b);
  std::cout<<id<<" "<<ra<<" "<<int(sa.instance_id)<<" "<<int(sa.reason)<<" "<<rb<<" "<<int(sb.instance_id)<<" "<<int(sb.reason)<<"\n";
  if(mode) {
   using Clock=std::chrono::steady_clock;
   for(int block=0;block<11;++block)for(int side=0;side<2;++side) {
    const int arm=(side+block)%2;auto start=Clock::now();
    for(int j=0;j<20000;++j) {
     ck_tile::GroupedGemmSelectorPolicy q{};int status=arm?read_policy_1(p,q,disabled):read_policy_0(p,q,disabled);
     asm volatile("" : : "g"(&q),"g"(status) : "memory");
    }
    auto stop=Clock::now();std::cout<<"T "<<id<<" "<<block<<" "<<arm<<" "<<std::chrono::duration<double,std::nano>(stop-start).count()/20000<<"\n";
   }
  }
 }
}
