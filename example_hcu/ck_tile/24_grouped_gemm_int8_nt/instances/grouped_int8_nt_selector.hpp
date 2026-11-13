// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include "grouped_int8_nt_routes.hpp"

namespace ck_tile {

// Rules derived from paired BF16/FP16 tuning on 80-CU gfx936. Call only for
// equal, nonempty groups with dense A/B/C and scalar-A/per-channel-B scales.
// LegacyRaw means no tuned override; retain the established selector result.
inline unsigned int8_cached_tuned_scaled_route(
    int m, int n, int k, int groups, bool packed, int compute_units)
{
    using namespace int8_routes;
    if(compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU ||
       m <= 0 || n <= 0 || k <= 0 || groups <= 0)
        return LegacyRaw;

    if(m == 1 && n >= 2048 && n <= 4608 && n % 256 == 0 &&
       k >= 4096 && k <= 6144 && k % 128 == 0)
        return KParallelDecode;

    if(m <= 8 && packed && n % 64 == 0 && k % 256 == 0)
    {
        // Short K favors direct VGPR loads, especially with fewer groups.
        // At N4096 the same route also wins on measured mid-K, small-G shapes.
        if((k == 512 && (groups <= 16 || n <= 256 || m <= 2)) ||
           (k == 768 && groups >= 24) ||
           (n == 4096 && k >= 1024 && k <= 2048 && groups <= 16))
            return PackedCooperativeM1N256;
    }

    if(m == 16 && n % 32 == 0 && k % 512 == 0 &&
       ((k >= 1024 && k <= 1536) || (k == 2048 && groups >= 24) ||
        (k == 4096 && groups <= 16)))
        return RawAlignedM16N32K512;

    if(m == 32)
    {
        if(packed && k <= 2048)
            return PackedM32N64;
        if(!packed && groups >= 128 && n % 4 == 0 && k >= 1024 && k <= 2048 &&
           k % 16 == 0)
            return RawM32N64K256W4;
    }

    if(packed && (m == 2048 || m == 4096) && n == 2048 && k == 512 &&
       groups >= 8 && groups <= 64)
        return PairedM128N256;

    // For four groups in the measured large-M range, the four-M-tile
    // schedule and cross-half A prefetch improve the paired-store path.
    if(packed && m >= 14336 && m <= 16384 && m % 2048 == 0 && n == 1536 &&
       k == 2048 && groups == 4)
        return PairedM256N384MaskedM4;

    // For M=8192, N=3072, K=4096 with G=4 or G=6, 2D MGroup=4 clustering
    // and cross-half A prefetch improve L2 hit rate and throughput over standard BN384.
    if(packed && m == 8192 && n == 3072 && k == 4096 && (groups == 4 || groups == 6))
        return PairedM256N384MaskedM4;
    return LegacyRaw;
}

// Calibrated on all 344 original M >= 256 requirements in iteration 96.
// Dimensions only; no case IDs, lookups or target workbook access at runtime.
inline unsigned int8_cached_base_route(int m, int n, int k, int groups, bool packed_available)
{
    // Only the measured low-group/long-K M32 family needs the packed path.
    if(m == 32)
    {
        if(!packed_available || k < 3072 || groups > 12)
            return int8_routes::LegacyRaw;
        return (n >= 4096 && k >= 4096) ? int8_routes::PackedM32N256 : int8_routes::PackedM32N64;
    }

    // M64 consumer stripe width depends on workload dimensions, never case IDs.
    if(m == 64)
    {
        if(!packed_available)
            return int8_routes::LegacyRaw;
        return (n >= 4096 && k >= 4096 && groups <= 12) ? int8_routes::PackedM64N256
                                                        : int8_routes::PackedM64N64;
    }

    // Two M128 packed variants, calibrated across all 422 M >= 128 requirements.
    // Only dimensions/availability are used; non-native plans are filtered above.
    if(m == 128)
        return packed_available ? int8_routes::PackedM128N128 : int8_routes::LegacyRaw;

    if(packed_available)
    {
        if(m == 256 &&
           ((n == 3072 && k >= 1536) || (k == 2048 && (n <= 1024 || (n <= 1536 && groups >= 16)))))
            return int8_routes::PackedM128N128;
        if(m >= 8192 && k == 512)
            return int8_routes::PackedM128N256;
        if(m >= 8192 && k == 768)
            return int8_routes::PackedM128N128;
        if(m >= 2048 && k == 2048 && n <= 1024)
            return int8_routes::PackedM128N256;
    }

    const bool power_of_two = (k & (k - 1)) == 0;
    if(m == 256 || k < 1024 || (k == 2048 && n <= 1024 && (m <= 4096 || groups <= 16)))
        return power_of_two ? int8_routes::LinearTransposedRaw
                            : int8_routes::LinearTransposedRecord1024;

    return packed_available ? int8_routes::PackedWeightM256N256 : int8_routes::LegacyRaw;
}

// Both candidate M256 kernels reserve 64KiB LDS: one CTA per CU on gfx936.
// Thresholds are measured scheduling-cost calibration, not hardware limits.
inline unsigned int8_cached_route(int m,
                                  int n,
                                  int k,
                                  int groups,
                                  bool packed_available,
                                  int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    const unsigned base = int8_cached_base_route(m, n, k, groups, packed_available);
    if(!packed_available || base != int8_routes::PackedWeightM256N256 || m < 2048 || k < 1024 ||
       compute_units <= 0)
        return base;

    const std::uint64_t mt         = (std::uint64_t(m) + 255u) / 256u;
    const std::uint64_t padded_m   = (mt + 3u) / 4u * 4u;
    const std::uint64_t old_ctas   = padded_m * ((std::uint64_t(n) + 255u) / 256u) * groups;
    const std::uint64_t wide_ctas  = padded_m * ((std::uint64_t(n) + 383u) / 384u) * groups;
    const std::uint64_t old_waves  = (old_ctas + compute_units - 1u) / compute_units;
    const std::uint64_t wide_waves = (wide_ctas + compute_units - 1u) / compute_units;

    // Evaluates whether wide tile N=384 scheduling delivers greater occupancy than standard N=256
    if(old_waves * 100u >= wide_waves * int8_tuning_config::WAVE_EFFICIENCY_RATIO_PCT ||
       (old_waves >= int8_tuning_config::HIGH_WAVE_COUNT_THRESHOLD &&
        old_waves * 100u >= wide_waves * int8_tuning_config::WAVE_HIGH_OCCUPANCY_RATIO_PCT))
        return int8_routes::PackedM256N384;

    return base;
}

// The prepared-B cooperative domain is independent of the WASP native-M domain.
// Calibrated on M1/2/4/8, G8 and the complete long-K requirement neighborhood.
inline unsigned int8_cached_scaled_small_route(
    int m,
    int n,
    int k,
    int groups,
    bool packed_available,
    int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    if(compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU)
        return int8_routes::LegacyRaw;

    // Raw NT K-parallel reduction; requires no prepared weight layout.
    if((m == 1 || m == 2) && n >= 4096 && n <= 6144 && n % 256 == 0 && k >= 7168 &&
       k <= 8192 && k % 128 == 0 &&
       (groups >= 12 || (groups == 8 && n == 4096 && k == 7168)))
        return int8_routes::KParallelDecode;

    // The caller has checked uniform, nonempty groups. All three tile
    // dimensions are exact multiples for the aligned M16 instance.
    if(m == 16 && n == 4096 && (k == 6144 || k == 7168) && groups == 8)
        return int8_routes::RawAlignedM16N32K512;
    // BN64 reduces the operand LDS allocation from 42.5 to 25.5 KiB
    // in this measured large-group domain.
    if(m == 32 && n == 4096 && k == 8192 && (groups == 64 || groups == 128))
        return int8_routes::RawM32N64K256W4;

    // Ultra-narrow M <= 8 decode专线: zero-LDS direct global memory bypass
    return packed_available && m > 0 && m <= 8 && n == 4096 && (k == 6144 || k == 7168) && groups == 8
               ? int8_routes::PackedCooperativeM1N256
               : int8_routes::LegacyRaw;
}

// Scaled outputs have a separate policy; the established INT32 route is unchanged.
// Formal paired measurements cover the low-N family and K768 M128 packed tile.
inline unsigned int8_cached_paired_store_route(
    unsigned route, int m, int n, int k, int groups, int compute_units, bool dense_scalar)
{
    if(!dense_scalar || compute_units != int8_tuning_config::GFX936_DEFAULT_NUM_CU || m < 2048 ||
       m > 16384 || n < 256 || n > 8192 || k < 512 || k > 8192 || groups < 4 || groups > 224)
        return route;

    // With four experts (and M=4096 G=8) this narrow family benefits from more M tiles.
    // Keep the larger M and other group counts on the existing WASP geometry.
    if((route == int8_routes::PackedWeightM256N256 || route == int8_routes::PackedM256N384) &&
       n == 1536 && k == 2048 && (groups == 4 || (groups == 8 && m == 4096)) && m <= 8192 &&
       (m & (m - 1)) == 0)
        return int8_routes::PairedM128N256;

    // Upgrade routes to paired-store epilogue implementations
    switch(route)
    {
    case int8_routes::PackedWeightM256N256:   return int8_routes::PairedM256N256;
    case int8_routes::PackedM128N256:         return int8_routes::PairedM128N256;
    case int8_routes::PackedM256N384:         return int8_routes::PairedM256N384;
    case int8_routes::PackedM256N384MaskedM4: return int8_routes::PairedM256N384MaskedM4;
    case int8_routes::PackedM256N384MaskedM8: return int8_routes::PairedM256N384MaskedM8;
    default: return route;
    }
}

inline unsigned int8_cached_scaled_route(
    int m,
    int n,
    int k,
    int groups,
    bool packed_available,
    int compute_units = int8_tuning_config::GFX936_DEFAULT_NUM_CU)
{
    if(packed_available && m >= 256 && n == 256 && k == 512)
        return m <= 1024 ? int8_routes::PackedM64N256 : int8_routes::PackedM128N256;
    if(packed_available && m >= 2048 && n == 2048 && k == 768)
        return int8_routes::PackedM128N256;

    // Calibrated across 156 original tail shapes on 80-CU gfx936. Keep complete
    // N blocks and unmeasured dimension ranges on their established policy.
    // Masking removes absent 32-column consumers; cl8 is preferable for a wide
    // tail (128 valid columns) or long K, while cl4 retains short K locality.
    if(packed_available && compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU &&
       m >= 2048 && m % 256 == 0 && n >= 3584 && n <= 8192 && n % 256 == 0 && n % 384 != 0 &&
       k >= 1024 && k < 16384 && k % 128 == 0)
        return k >= 4096 || n % 384 == 128 ? int8_routes::PackedM256N384MaskedM8
                                           : int8_routes::PackedM256N384MaskedM4;

    // Full BN384 blocks: calibrate the scheduling threshold in the measured
    // N4608/K5120 neighborhood. Use CTA batches, never workload IDs.
    const unsigned route = int8_cached_route(m, n, k, groups, packed_available, compute_units);
    if(route == int8_routes::PackedWeightM256N256 && packed_available &&
       compute_units == int8_tuning_config::GFX936_DEFAULT_NUM_CU && m >= 2048 && m <= 4096 &&
       m % 256 == 0 && n == 4608 && k == 5120)
    {
        const std::uint64_t mt       = (std::uint64_t(m) + 255u) / 256u;
        const std::uint64_t padded_m = (mt + 3u) / 4u * 4u;
        const std::uint64_t old_batches =
            (padded_m * ((std::uint64_t(n) + 255u) / 256u) * groups + 79u) / 80u;
        const std::uint64_t wide_batches =
            (padded_m * (std::uint64_t(n) / 384u) * groups + 79u) / 80u;
        if(old_batches >= int8_tuning_config::HIGH_BATCH_COUNT_THRESHOLD &&
           old_batches * 100u >= wide_batches * int8_tuning_config::BATCH_EFFICIENCY_RATIO_PCT)
            return int8_routes::PackedM256N384;
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
};

inline unsigned int8_cached_select_plan_route(int m,
                                             int n,
                                             int k,
                                             int groups,
                                             bool packed,
                                             int compute_units,
                                             bool scaled,
                                             const Int8CachedPlanRouteDomain& domain)
{
    using namespace int8_routes;
    if(scaled && domain.uniform_nonempty)
    {
        if(domain.dense_channel_scales)
        {
            const auto route =
                int8_cached_tuned_scaled_route(m, n, k, groups, packed, compute_units);
            if(route != LegacyRaw)
                return route;
        }
        const auto route = int8_cached_scaled_small_route(m, n, k, groups, packed, compute_units);
        if(route != LegacyRaw)
            return route;
    }

    if(!domain.native_wasp)
        return LegacyRaw;
    const auto route = scaled ? int8_cached_scaled_route(m, n, k, groups, packed, compute_units)
                              : int8_cached_route(m, n, k, groups, packed, compute_units);
    if(!domain.raw_transposed &&
       (route == LinearTransposedRaw || route == LinearTransposedRecord1024))
        return LegacyRaw;
    return scaled ? int8_cached_paired_store_route(
                        route, m, n, k, groups, compute_units, domain.dense_scalar_a)
                  : route;
}

} // namespace ck_tile
