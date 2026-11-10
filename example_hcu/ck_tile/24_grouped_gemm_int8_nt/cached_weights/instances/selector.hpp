// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace ck_tile {
// Calibrated on all344 originalM>=256 requirements in iteration96.
// Dimensions only;no case IDs,lookups or target workbook access at runtime.
inline unsigned int8_cached_base_route(int m, int n, int k, int groups, bool packed_available)
{
    // Only the measured low-group/longK M32 family needs the packed path.
    if(m == 32)
    {
        if(!packed_available || k < 3072 || groups > 12)
            return 0u;
        return n >= 4096 && k >= 4096 ? 10u : 11u;
    }
    // M64 consumer stripe width depends on workload dimensions,never case IDs.
    if(m == 64)
    {
        if(!packed_available)
            return 0u;
        return n >= 4096 && k >= 4096 && groups <= 12 ? 8u : 9u;
    }
    // Two M128 packed variants,calibrated across all422 M>=128 requirements.
    // Only dimensions/availability are used;non-native plans are filtered above.
    if(m == 128)
        return packed_available ? 7u : 0u;
    if(packed_available)
    {
        if(m == 256 &&
           ((n == 3072 && k >= 1536) || (k == 2048 && (n <= 1024 || (n <= 1536 && groups >= 16)))))
            return 7u;
        if(m >= 8192 && k == 512)
            return 6u;
        if(m >= 8192 && k == 768)
            return 7u;
        if(m >= 2048 && k == 2048 && n <= 1024)
            return 6u;
    }
    const bool power_of_two = (k & (k - 1)) == 0;
    if(m == 256 || k < 1024 || (k == 2048 && n <= 1024 && (m <= 4096 || groups <= 16)))
        return power_of_two ? 3u : 4u;
    return packed_available ? 5u : 0u;
}
// Both candidate M256 kernels reserve64KiB LDS:one CTA perCU on gfx936.
// Thresholds are measured scheduling-cost calibration,not hardware limits.
inline unsigned
int8_cached_route(int m, int n, int k, int groups, bool packed_available, int compute_units = 80)
{
    const unsigned base = int8_cached_base_route(m, n, k, groups, packed_available);
    if(!packed_available || base != 5u || m < 2048 || k < 1024 ||
       compute_units <= 0)
        return base;
    const std::uint64_t mt         = (std::uint64_t(m) + 255u) / 256u;
    const std::uint64_t padded_m   = (mt + 3u) / 4u * 4u;
    const std::uint64_t old_ctas   = padded_m * ((std::uint64_t(n) + 255u) / 256u) * groups;
    const std::uint64_t wide_ctas  = padded_m * ((std::uint64_t(n) + 383u) / 384u) * groups;
    const std::uint64_t old_waves  = (old_ctas + compute_units - 1u) / compute_units;
    const std::uint64_t wide_waves = (wide_ctas + compute_units - 1u) / compute_units;
    if(old_waves * 100u >= wide_waves * 147u ||
       (old_waves >= 24u && old_waves * 100u >= wide_waves * 144u))
        return 12u;
    return base;
}

// The prepared-B cooperative domain is independent of the WASP native-M domain.
// Calibrated on M1/2/4/8, G8 and the complete long-K requirement neighborhood.
inline unsigned int8_cached_scaled_small_route(
    int m, int n, int k, int groups, bool packed_available, int compute_units = 80)
{
    // Raw NT K-parallel reduction;requires no prepared weight layout.
    if(compute_units == 80 && (m == 1 || m == 2) && groups >= 12 && n >= 4096 && n <= 6144 &&
       n % 256 == 0 && k >= 7168 && k <= 8192 && k % 128 == 0)
        return 25u;
    return packed_available && compute_units == 80 && m > 0 && m <= 8 && n == 4096 &&
                   (k == 6144 || k == 7168) && groups == 8
               ? 24u
               : 0u;
}

// Scaled outputs have a separate policy;the established INT32 route is unchanged.
// Formal paired measurements cover the low-N family and K768 M128 packed tile.
inline unsigned int8_cached_paired_store_route(
    unsigned route, int m, int n, int k, int groups, int compute_units, bool dense_scalar)
{
    if(!dense_scalar || compute_units != 80 || m < 2048 || m > 16384 || n < 1536 || n > 8192 ||
       k < 1024 || k > 8192 || groups < 4 || groups > 224)
        return route;
    // With four experts (and M=4096 G=8) this narrow family benefits from more M tiles.
    // Keep the larger M and other group counts on the existing WASP geometry.
    if((route == 5u || route == 12u) && n == 1536 && k == 2048 &&
       (groups == 4 || (groups == 8 && m == 4096)) && m <= 8192 && (m & (m - 1)) == 0)
        return 30u;
    switch(route)
    {
    case 5: return 26;
    case 12: return 27;
    case 22: return 28;
    case 23: return 29;
    default: return route;
    }
}

inline unsigned int8_cached_scaled_route(
    int m, int n, int k, int groups, bool packed_available, int compute_units = 80)
{
    if(packed_available && m >= 256 && n == 256 && k == 512)
        return m <= 1024 ? 8u : 6u;
    if(packed_available && m >= 2048 && n == 2048 && k == 768)
        return 6u;
    // Calibrated across156 original tail shapes on80-CU gfx936. Keep complete
    // N blocks and unmeasured dimension ranges on their established policy.
    // Masking removes absent32-column consumers; cl8 is preferable for a wide
    // tail (128 valid columns) or longK, while cl4 retains shortK locality.
    if(packed_available && compute_units == 80 && m >= 2048 && m % 256 == 0 && n >= 3584 &&
       n <= 8192 && n % 256 == 0 && n % 384 != 0 && k >= 1024 && k < 16384 && k % 128 == 0)
        return k >= 4096 || n % 384 == 128 ? 23u : 22u;
    // Full BN384 blocks: calibrate the scheduling threshold in the measured
    // N4608/K5120 neighborhood. Use CTA batches, never workload IDs.
    const unsigned route = int8_cached_route(m, n, k, groups, packed_available, compute_units);
    if(route == 5u && packed_available && compute_units == 80 && m >= 2048 && m <= 4096 &&
       m % 256 == 0 && n == 4608 && k == 5120)
    {
        const std::uint64_t mt       = (std::uint64_t(m) + 255u) / 256u;
        const std::uint64_t padded_m = (mt + 3u) / 4u * 4u;
        const std::uint64_t old_batches =
            (padded_m * ((std::uint64_t(n) + 255u) / 256u) * groups + 79u) / 80u;
        const std::uint64_t wide_batches =
            (padded_m * (std::uint64_t(n) / 384u) * groups + 79u) / 80u;
        if(old_batches >= 20u && old_batches * 100u >= wide_batches * 146u)
            return 12u;
    }
    return route;
}

} // namespace ck_tile
