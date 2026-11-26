// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "grouped_int8_nt_routes.hpp"

namespace ck_tile {

// Shape envelope used to tune the vectorized NT routes on gfx936.
inline bool int8_cached_vector_nt_domain(int n, int k)
{
    return n % 256 == 0 && k % 128 == 0 && k >= 512 && k < 16384;
}

// The four-wave cooperative instance skips operand and epilogue bounds checks.
inline bool int8_cached_raw_w4_exact_tile(int n, int k)
{
    constexpr int block_n = 64, block_k = 256;
    return n % block_n == 0 && k % block_k == 0;
}

// Rules derived from paired BF16/FP16 tuning on 80-CU gfx936. Call only for
// equal, nonempty groups with dense A/B/C and scalar-A/per-channel-B scales.
// LegacyRaw means no tuned override; retain the established selector result.
inline unsigned
int8_cached_tuned_scaled_route(int m,
                            int n,
                            int k,
                            int groups,
                            bool vector_nt_domain,
                            int compute_units,
                            bool fp16_output = false)
{
    using namespace int8_routes;
    if(compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU || m <= 0 || n <= 0 || k <= 0 ||
       groups <= 0)
        return LegacyRaw;

    if(m == 1 && n >= 2048 && n <= 4608 && n % 256 == 0 && k >= 4096 && k <= 6144 && k % 128 == 0)
        return KParallelDecode;

    if(m <= 8 && vector_nt_domain && n % 64 == 0 && k % 256 == 0)
    {
        // Short K favors the narrow cooperative tile in this shape range.
        // At N4096 the same route also wins on measured mid-K, small-G shapes.
        if((k == 512 && (groups <= 16 || n <= 256 || m <= 2)) || (k == 768 && groups >= 24) ||
           (n == 4096 && k >= 1024 && k <= 2048 && groups <= 16))
            return RawSmallM16N32K512;
    }

    if(m == 16 && n % 32 == 0 && k % 512 == 0 &&
       ((k >= 1024 && k <= 1536) || (k == 2048 && groups >= 24) || (k == 4096 && groups <= 16)))
        return RawAlignedM16N32K512;

    if(m == 32)
    {
        if(vector_nt_domain && k == 2048 && n >= 7168 && groups >= 8 && groups <= 128 &&
           int8_cached_raw_w4_exact_tile(n, k))
            return RawM32N64K256W4;
        if(vector_nt_domain && k <= 2048)
            return NativeM32N64;
        if(!vector_nt_domain && groups >= 128 && k >= 1024 && k <= 2048 &&
           int8_cached_raw_w4_exact_tile(n, k))
            return RawM32N64K256W4;
    }

    // A single M tile and eight full N tiles use the BN384 paired-store kernel.
    // Both output types use the same raw-weight kernel in the measured group range.
    if(vector_nt_domain && m == 256 && n == 3072 && k == 1536 && groups >= 8 && groups <= 64)
        return PairedM256N384MaskedM4;

    if(vector_nt_domain && (m == 2048 || m == 4096) && n == 2048 && k == 512 && groups >= 8 && groups <= 64)
        return PairedM128N256;

    // For four groups in the measured large-M range, the four-M-tile
    // schedule and cross-half A prefetch improve the paired-store path.
    if(vector_nt_domain && m >= 14336 && m <= 16384 && m % 2048 == 0 && n == 1536 && k == 2048 && groups == 4)
        return m == 16384 ? PairedM256N384MaskedM4Lds : PairedM256N384MaskedM4;

    // Grouped-M traversal wins with eight full N tiles at the measured group counts.
    // At six groups the LDS epilogue gives FP16 more headroom than register writeback.
    if(vector_nt_domain && n == 3072 && k == 4096)
    {
        if(m == 8192 && (groups == 4 || groups == 6 || groups == 8 || groups == 12))
            return fp16_output && groups == 6 ? PairedM256N384MaskedM4Lds
                                              : PairedM256N384MaskedM4;
        if(m == 16384 && (groups == 4 || groups == 6 || groups == 8))
            return PairedM256N384MaskedM4;
    }
    return LegacyRaw;
}

// Calibrated on all 344 original M >= 256 requirements in iteration 96.
// Dimensions only; no case IDs, lookups or target workbook access at runtime.
inline unsigned int8_cached_base_route(int m, int n, int k, int groups, bool vector_nt_domain)
{
    // Only the measured low-group/long-K M32 family uses the tuned NT WASP route.
    if(m == 32)
    {
        if(!vector_nt_domain || k < 3072 || groups > 12)
            return int8_routes::LegacyRaw;
        return (n >= 4096 && k >= 4096) ? int8_routes::NativeM32N256 : int8_routes::NativeM32N64;
    }

    // M64 consumer stripe width depends on workload dimensions, never case IDs.
    if(m == 64)
    {
        if(!vector_nt_domain)
            return int8_routes::LegacyRaw;
        return (n >= 4096 && k >= 4096 && groups <= 12) ? int8_routes::NativeM64N256
                                                        : int8_routes::NativeM64N64;
    }

    // Two M128 NT variants, calibrated across all 422 M >= 128 requirements.
    // Only dimensions/availability are used; non-native plans are filtered above.
    if(m == 128)
        return vector_nt_domain ? int8_routes::NativeM128N128 : int8_routes::LegacyRaw;

    if(vector_nt_domain)
    {
        if(m == 256 &&
           ((n == 3072 && k >= 1536) || (k == 2048 && (n <= 1024 || (n <= 1536 && groups >= 16)))))
            return int8_routes::NativeM128N128;
        if(m >= 8192 && k == 512)
            return int8_routes::NativeM128N256;
        if(m >= 8192 && k == 768)
            return int8_routes::NativeM128N128;
        if(m >= 2048 && k == 2048 && n <= 1024)
            return int8_routes::NativeM128N256;
    }

    const bool power_of_two = (k & (k - 1)) == 0;
    if(m == 256 || k < 1024 || (k == 2048 && n <= 1024 && (m <= 4096 || groups <= 16)))
        return power_of_two ? int8_routes::LinearTransposedRaw
                            : int8_routes::LinearTransposedRecord1024;

    return vector_nt_domain ? int8_routes::NativeM256N256 : int8_routes::LegacyRaw;
}

// Both candidate M256 kernels reserve 64KiB LDS: one CTA per CU on gfx936.
// Thresholds are measured scheduling-cost calibration, not hardware limits.
inline unsigned int8_cached_route(int m,
                                  int n,
                                  int k,
                                  int groups,
                                  bool vector_nt_domain,
                                  int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    const unsigned base = int8_cached_base_route(m, n, k, groups, vector_nt_domain);
    if(!vector_nt_domain || base != int8_routes::NativeM256N256 || m < 2048 || k < 1024 ||
       compute_units <= 0)
        return base;

    constexpr std::uint64_t tile_m          = 256;
    constexpr std::uint64_t standard_tile_n = 256;
    constexpr std::uint64_t wide_tile_n     = 384;
    constexpr std::uint64_t m_cluster       = 4;
    const std::uint64_t mt                  = (std::uint64_t(m) + tile_m - 1u) / tile_m;
    const std::uint64_t padded_m            = (mt + m_cluster - 1u) / m_cluster * m_cluster;
    const std::uint64_t old_ctas =
        padded_m * ((std::uint64_t(n) + standard_tile_n - 1u) / standard_tile_n) * groups;
    const std::uint64_t wide_ctas =
        padded_m * ((std::uint64_t(n) + wide_tile_n - 1u) / wide_tile_n) * groups;
    const std::uint64_t standard_batches = (old_ctas + compute_units - 1u) / compute_units;
    const std::uint64_t wide_batches     = (wide_ctas + compute_units - 1u) / compute_units;

    // Compare scheduled CU batches for BN384 and BN256. Both use one CTA per CU.
    if(standard_batches * 100u >= wide_batches * int8_tuning_config::WAVE_EFFICIENCY_RATIO_PCT ||
       (standard_batches >= int8_tuning_config::HIGH_WAVE_COUNT_THRESHOLD &&
        standard_batches * 100u >=
            wide_batches * int8_tuning_config::WAVE_HIGH_OCCUPANCY_RATIO_PCT))
        return int8_routes::NativeM256N384;

    return base;
}

// The small-M cooperative domain is independent of the WASP native-M domain.
// Calibrated on M1/2/4/8, G8 and the complete long-K requirement neighborhood.
inline unsigned
int8_cached_scaled_small_route(int m,
                               int n,
                               int k,
                               int groups,
                               bool vector_nt_domain,
                               int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    if(compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU)
        return int8_routes::LegacyRaw;

    // Raw NT K-parallel reduction; requires no prepared weight layout.
    if((m == 1 || m == 2) && n >= 4096 && n <= 6144 && n % 256 == 0 && k >= 7168 && k <= 8192 &&
       k % 128 == 0 && (groups >= 12 || (groups == 8 && n == 4096 && k == 7168)))
        return int8_routes::KParallelDecode;

    // The caller has checked uniform, nonempty groups. All three tile
    // dimensions are exact multiples for the aligned M16 instance.
    if(m == 16 && n == 4096 && (k == 6144 || k == 7168) && groups == 8)
        return int8_routes::RawAlignedM16N32K512;
    // BN64 reduces the operand LDS allocation from 42.5 to 25.5 KiB
    // in this measured large-group domain.
    if(m == 32 && n == 4096 && k == 8192 && (groups == 64 || groups == 128))
        return int8_routes::RawM32N64K256W4;

    // Use the small-M cooperative tile for this low-group, long-K range.
    return vector_nt_domain && m > 0 && m <= 8 && n == 4096 && (k == 6144 || k == 7168) &&
                   groups == 8
               ? int8_routes::RawSmallM16N32K512
               : int8_routes::LegacyRaw;
}

// Formal paired measurements cover the low-N family and K768 M128 NT tile.
inline unsigned int8_cached_paired_store_route(
    unsigned route, int m, int n, int k, int groups, int compute_units, bool dense_scalar)
{
    if(!dense_scalar || compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU || m < 2048 ||
       m > 16384 || n < 256 || n > 8192 || k < 512 || k > 8192 || groups < 4 || groups > 224)
        return route;

    // With four experts (and M=4096 G=8) this narrow family benefits from more M tiles.
    // Keep the larger M and other group counts on the existing WASP geometry.
    if((route == int8_routes::NativeM256N256 || route == int8_routes::NativeM256N384) &&
       n == 1536 && k == 2048 && (groups == 4 || (groups == 8 && m == 4096)) && m <= 8192 &&
       (m & (m - 1)) == 0)
        return int8_routes::PairedM128N256;

    // Upgrade routes to paired-store epilogue implementations
    switch(route)
    {
    case int8_routes::NativeM256N256: return int8_routes::PairedM256N256;
    case int8_routes::NativeM128N256: return int8_routes::PairedM128N256;
    case int8_routes::NativeM256N384: return int8_routes::PairedM256N384;
    case int8_routes::NativeM256N384MaskedM4: return int8_routes::PairedM256N384MaskedM4;
    case int8_routes::NativeM256N384MaskedM8: return int8_routes::PairedM256N384MaskedM8;
    default: return route;
    }
}

inline unsigned
int8_cached_scaled_route(int m,
                         int n,
                         int k,
                         int groups,
                         bool vector_nt_domain,
                         int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    if(vector_nt_domain && m >= 256 && n == 256 && k == 512)
        return m <= 1024 ? int8_routes::NativeM64N256 : int8_routes::NativeM128N256;
    if(vector_nt_domain && m >= 2048 && n == 2048 && k == 768)
        return int8_routes::NativeM128N256;

    // Calibrated across 156 original tail shapes on 80-CU gfx936. Keep complete
    // N blocks and unmeasured dimension ranges on their established policy.
    // Masking removes absent 32-column consumers; cl8 is preferable for a wide
    // tail (128 valid columns) or long K, while cl4 retains short K locality.
    if(vector_nt_domain && compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU &&
       m >= 2048 && m % 256 == 0 && n >= 3584 && n <= 8192 && n % 256 == 0 && n % 384 != 0 &&
       k >= 1024 && k < 16384 && k % 128 == 0)
        return k >= 4096 || n % 384 == 128 ? int8_routes::NativeM256N384MaskedM8
                                           : int8_routes::NativeM256N384MaskedM4;

    // Full BN384 blocks: calibrate the scheduling threshold in the measured
    // N4608/K5120 neighborhood. Use CTA batches, never workload IDs.
    const unsigned route = int8_cached_route(m, n, k, groups, vector_nt_domain, compute_units);
    if(route == int8_routes::NativeM256N256 && vector_nt_domain &&
       compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU && m >= 2048 && m <= 4096 &&
       m % 256 == 0 && n == 4608 && k == 5120)
    {
        constexpr std::uint64_t tile_m          = 256;
        constexpr std::uint64_t standard_tile_n = 256;
        constexpr std::uint64_t wide_tile_n     = 384;
        constexpr std::uint64_t m_cluster       = 4;
        const auto cu_count                     = std::uint64_t(compute_units);
        const std::uint64_t mt                  = (std::uint64_t(m) + tile_m - 1u) / tile_m;
        const std::uint64_t padded_m            = (mt + m_cluster - 1u) / m_cluster * m_cluster;
        const std::uint64_t old_batches =
            (padded_m * ((std::uint64_t(n) + standard_tile_n - 1u) / standard_tile_n) * groups +
             cu_count - 1u) /
            cu_count;
        const std::uint64_t wide_batches =
            (padded_m * (std::uint64_t(n) / wide_tile_n) * groups + cu_count - 1u) / cu_count;
        if(old_batches >= int8_tuning_config::HIGH_BATCH_COUNT_THRESHOLD &&
           old_batches * 100u >= wide_batches * int8_tuning_config::BATCH_EFFICIENCY_RATIO_PCT)
            return int8_routes::NativeM256N384;
    }
    return route;
}

struct Int8CachedPlanRouteDomain
{
    // The general cached WASP domain and the small-M domain are independent.
    bool native_wasp;
    bool uniform_nonempty;
    bool raw_transposed;
    bool dense_scalar_a;
    bool dense_channel_scales;
    bool fp16_output = false;
};

inline unsigned
int8_cached_select_plan_route(int m,
                              int n,
                              int k,
                              int groups,
                              bool vector_nt_domain,
                              int compute_units,
                              const Int8CachedPlanRouteDomain& domain,
                              GroupedGemmInstanceId fallback = GroupedGemmInstanceId::none)
{
    using namespace int8_routes;
    if(domain.uniform_nonempty)
    {
        if(domain.dense_channel_scales)
        {
            const auto route =
                int8_cached_tuned_scaled_route(
                     m, n, k, groups, vector_nt_domain, compute_units, domain.fp16_output);
            if(route != LegacyRaw)
                return route;
        }
        const auto route = int8_cached_scaled_small_route(m, n, k, groups, vector_nt_domain, compute_units);
        if(route != LegacyRaw)
            return route;
    }

    const auto route = domain.native_wasp
                           ? int8_cached_scaled_route(m, n, k, groups, vector_nt_domain, compute_units)
                           : LegacyRaw;
    if(compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU && domain.uniform_nonempty &&
       domain.dense_channel_scales)
    {
        if(m == 32 && int8_cached_raw_w4_exact_tile(n, k))
        {
            if(k == 2048 && n >= 7168 && groups >= 8 && groups <= 96)
                return RawM32N64K256W4;
            if(n == 4096 && k == 7168 && (groups == 12 || groups == 24))
                return LegacyRaw;
            if(n >= 2048 && n <= 7168 && k >= 3072 && k <= 7168 && groups >= 6 && groups <= 32 &&
               !(n == 7168 && k == 3072 && groups >= 24))
                return RawM32N64K256W4;
        }
        if(m == 64 && ((n == 7168 && k >= 2048) || (n >= 4096 && k >= 4096)))
            return LegacyRaw;
        if(m == 128 && vector_nt_domain && domain.dense_scalar_a && n == 4096 && k == 4096 &&
           groups == 8)
            return PairedM128N256;
        if(m == 256 && vector_nt_domain && n == 3072 && k >= 3072)
        {
            if(k == 4096 && (groups == 8 || groups == 16 || groups == 48) && domain.dense_scalar_a)
                return PairedM256N384;
            return domain.raw_transposed
                       ? ((k & (k - 1)) == 0 ? LinearTransposedRaw : LinearTransposedRecord1024)
                       : LegacyRaw;
        }
        if(m == 256 && vector_nt_domain && n == 4096 && k >= 6144)
        {
            if(k == 7168 && groups == 12 && domain.dense_scalar_a)
                return PairedM256N384;
            if(groups == 8)
                return domain.raw_transposed ? LinearTransposedRecord1024 : LegacyRaw;
        }
        if(vector_nt_domain && domain.dense_scalar_a)
        {
            if(m == 4096 && n == 1024 && k == 2048 && groups == 8)
                return PairedM256N384MaskedM4;
            if(m == 8192 && n == 1536 && k == 2048 && groups == 4)
                return PairedM256N384MaskedM8;
            if(m == 4096 && n == 6144 && k == 7168)
                return PairedM256N384MaskedM8;
            if(m == 16384 && n == 4096 && k == 7168)
                return PairedM256N384MaskedM16;
        }

        // The four-wave BN64 tile is faster than the two-wave fallback in
        // the measured long-K, moderate-group M32 domain.
        if(route == LegacyRaw && fallback == GroupedGemmInstanceId::int8_nt_m32_n64_k256 &&
           m == 32 && n >= 2048 && n <= 7168 && k >= 3072 && k <= 7168 && groups >= 16 &&
           groups <= 32 && int8_cached_raw_w4_exact_tile(n, k))
            return RawM32N64K256W4;

        // Use the native NT paired-store tile inside the tuned vector shape domain.
        if(route == LinearTransposedRecord1024 && vector_nt_domain && m == 256 && n >= 2048 && n <= 4096 &&
           k >= 768 && k <= 7168 && groups >= 4 && groups <= 48)
            return PairedM128N256;
    }
    if(!domain.raw_transposed &&
       (route == LinearTransposedRaw || route == LinearTransposedRecord1024))
        return LegacyRaw;
    return int8_cached_paired_store_route(
        route, m, n, k, groups, compute_units, domain.dense_scalar_a);
}

struct Int8CachedPlanSelection
{
    unsigned route;
    GroupedGemmInstanceId legacy_instance;
};

// Selected by paired timing at both aligned and half-cache-line output offsets.
// Keep this policy scoped to the measured M/N/K/G families. The scaled
// direct-M16 entry and small-M K128 entry share the same kernel.
inline GroupedGemmInstanceId
int8_cached_tuned_m4_m8_instance(int m, int n, int k, int groups)
{
    using Instance = GroupedGemmInstanceId;
    const auto small_k128 = Instance::int8_nt_small_m_n64_k128;
    const auto small_k256 = Instance::int8_nt_small_m_n64_k256;
    const auto wide_k256  = Instance::int8_nt_small_m_n128_k256;
    const auto small_n32_k512 = Instance::int8_nt_small_g_n32_k512;

    switch(n)
    {
    case 256:
        if(k == 512 && groups == 64)
            return small_k128;
        break;
    case 1024:
        if(k == 2048 && groups == 32)
            return small_n32_k512;
        break;
    case 1536:
        if(k == 2048 && groups == 8)
            return small_k128;
        break;
    case 2048:
        if((k == 512 && (groups == 8 || groups == 16)) || (k == 768 && groups == 32))
            return small_k128;
        if(k == 768 && groups == 4)
            return small_k256;
        break;
    case 3072:
        if(k == 1232 && (groups == 196 || groups == 392))
            return small_k256;
        if(k == 3072 && groups == 16)
            return wide_k256;
        if(k == 3072 && groups == 8)
            return small_n32_k512;
        break;
    case 4096:
        if(k == 1024 && groups == 16)
            return small_k256;
        if(k == 1536)
        {
            if(groups == 4)
                return m == 8 ? small_k256 : small_k128;
            if(groups == 6 || groups == 8 || groups == 12 || groups == 16)
                return small_k256;
        }
        break;
    case 5120:
        if(k == 2304 && groups == 12)
            return wide_k256;
        break;
    case 7168:
        if(m == 4 && k == 3072 && groups == 12)
            return small_n32_k512;
        break;
    }
    return Instance::none;
}

inline Int8CachedPlanSelection int8_cached_select_plan(const GroupedGemmProblem& problem,
                                                        int max_m,
                                                        int compute_units,
                                                        const Int8CachedPlanRouteDomain& domain)
{
    using Instance = GroupedGemmInstanceId;
    using namespace int8_routes;
    const int n = problem.n.extent;
    const int k = problem.k.extent;
    const int groups = problem.group_count;
    const auto fallback = select_grouped_gemm_int8_nt_instance(problem);
    if((max_m == 4 || max_m == 8) &&
       compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU &&
       domain.uniform_nonempty && domain.dense_channel_scales)
    {
        const auto tuned =
            int8_cached_tuned_m4_m8_instance(max_m, n, k, groups);
        // The legacy small-G instance and Route 24 use the same scaled kernel.
        if(tuned == Instance::int8_nt_small_g_n32_k512)
            return {RawSmallM16N32K512, fallback};
        if(tuned != Instance::none)
            return {LegacyRaw, tuned};
    }

    return {int8_cached_select_plan_route(max_m,
                                          n,
                                          k,
                                          groups,
                                          int8_cached_vector_nt_domain(n, k),
                                          compute_units,
                                          domain,
                                          fallback),
            fallback};
}

} // namespace ck_tile
