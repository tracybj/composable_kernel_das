// SPDX-License-Identifier: MIT
#pragma once
namespace ck_tile {
// Calibrated on all344 originalM>=256 requirements in iteration96.
// Dimensions only;no case IDs,lookups or target workbook access at runtime.
inline unsigned int8_cached_route(int m,int n,int k,int groups,bool packed_available){
 const bool power_of_two=(k&(k-1))==0;
 if(m==256 || k<1024 || (k==2048 && n<=1024 && (m<=4096 || groups<=16)))
  return power_of_two?3u:4u;
 if(k==2048 && n>1024 && n<2048 && m>=4096)return 1u;
 return packed_available?5u:0u;
}
}
