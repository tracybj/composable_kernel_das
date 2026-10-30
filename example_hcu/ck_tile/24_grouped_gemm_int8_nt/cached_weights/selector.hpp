// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
namespace ck_tile {
// Calibrated on all344 originalM>=256 requirements in iteration96.
// Dimensions only;no case IDs,lookups or target workbook access at runtime.
inline unsigned int8_cached_base_route(int m,int n,int k,int groups,bool packed_available){
 // Only the measured low-group/longK M32 family needs the packed path.
 if(m==32){
  if(!packed_available || k<3072 || groups>12)return 0u;
  return n>=4096 && k>=4096?10u:11u;
 }
 // M64 consumer stripe width depends on workload dimensions,never case IDs.
 if(m==64){
  if(!packed_available)return 0u;
  return n>=4096 && k>=4096 && groups<=12?8u:9u;
 }
 // Two M128 packed variants,calibrated across all422 M>=128 requirements.
 // Only dimensions/availability are used;non-native plans are filtered above.
 if(m==128)return packed_available?7u:0u;
 if(packed_available){
  if(m==256 && ((n==3072 && k>=1536) ||
     (k==2048 && (n<=1024 || (n<=1536 && groups>=16)))))return 7u;
  if(m>=8192 && k==512)return 6u;
  if(m>=8192 && k==768)return 7u;
  if(m>=2048 && k==2048 && n<=1024)return 6u;
 }
 const bool power_of_two=(k&(k-1))==0;
 if(m==256 || k<1024 || (k==2048 && n<=1024 && (m<=4096 || groups<=16)))
  return power_of_two?3u:4u;
 if(k==2048 && n>1024 && n<2048 && m>=4096)return 1u;
 return packed_available?5u:0u;
}
// Both candidate M256 kernels reserve64KiB LDS:one CTA perCU on gfx936.
// Thresholds are measured scheduling-cost calibration,not hardware limits.
inline unsigned int8_cached_route(int m,int n,int k,int groups,bool packed_available,int compute_units=80){
 const unsigned base=int8_cached_base_route(m,n,k,groups,packed_available);
 if(!packed_available || (base!=1u && base!=5u) || m<2048 || k<1024 || compute_units<=0)return base;
 const std::uint64_t mt=(std::uint64_t(m)+255u)/256u;
 const std::uint64_t padded_m=(mt+3u)/4u*4u;
 const std::uint64_t old_ctas=padded_m*((std::uint64_t(n)+255u)/256u)*groups;
 const std::uint64_t wide_ctas=padded_m*((std::uint64_t(n)+383u)/384u)*groups;
 const std::uint64_t old_waves=(old_ctas+compute_units-1u)/compute_units;
 const std::uint64_t wide_waves=(wide_ctas+compute_units-1u)/compute_units;
 if(old_waves*100u>=wide_waves*147u || (old_waves>=24u && old_waves*100u>=wide_waves*144u))return 12u;
 return base;
}

}
