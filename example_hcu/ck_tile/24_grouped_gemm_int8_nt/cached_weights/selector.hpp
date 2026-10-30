// SPDX-License-Identifier: MIT
#pragma once
namespace ck_tile {
// Calibrated on all344 originalM>=256 requirements in iteration96.
// Dimensions only;no case IDs,lookups or target workbook access at runtime.
inline unsigned int8_cached_route(int m,int n,int k,int groups,bool packed_available){
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
}
