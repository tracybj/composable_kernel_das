#include "selector.hpp"
#include <cassert>
#include <array>
#include <iostream>
int main(){using ck_tile::int8_cached_route;
 assert(int8_cached_route(2048,1536,2048,4,true,80)==12);
 assert(int8_cached_route(2048,1536,2048,4,true,96)==5);
 assert(int8_cached_route(2048,3072,4096,4,true,80)==5);
 assert(int8_cached_route(2048,3072,4096,6,true,80)==12);
 assert(int8_cached_route(8192,1536,2048,4,true,80)==1);
 assert(int8_cached_route(16384,4096,7168,8,true,80)==12);
 assert(int8_cached_route(32,3072,4096,4,true,80)==11);
 assert(int8_cached_route(32,4096,7168,8,true,80)==10);
 assert(int8_cached_route(32,4096,7168,16,true,80)==0);
 assert(int8_cached_route(16384,4096,7168,8,false,80)==0);
 assert(int8_cached_route(2048,1536,2048,4,true,0)==5);
 using ck_tile::int8_cached_scaled_route;
 assert(int8_cached_scaled_route(256,256,512,64,true,80)==8);
 assert(int8_cached_scaled_route(1024,256,512,64,true,80)==8);
 assert(int8_cached_scaled_route(2048,256,512,64,true,80)==6);
 assert(int8_cached_scaled_route(16384,256,512,64,true,80)==6);
 assert(int8_cached_route(256,256,512,64,true,80)==3);
 assert(int8_cached_scaled_route(128,256,512,64,true,80)==7);
 assert(int8_cached_scaled_route(2048,2048,768,4,true,80)==6);
 assert(int8_cached_scaled_route(16384,2048,768,32,true,80)==6);
 assert(int8_cached_route(2048,2048,768,4,true,80)==4);
 assert(int8_cached_route(16384,2048,768,32,true,80)==7);
 assert(int8_cached_scaled_route(1024,2048,768,4,true,80)==int8_cached_route(1024,2048,768,4,true,80));
 assert(int8_cached_scaled_route(2048,2048,768,4,false,80)==int8_cached_route(2048,2048,768,4,false,80));
 assert(int8_cached_scaled_route(2048,1024,768,4,true,80)==int8_cached_route(2048,1024,768,4,true,80));
 assert(int8_cached_scaled_route(2048,4096,1024,128,true,80)==22);
 assert(int8_cached_scaled_route(2048,4096,1536,4,true,80)==22);
 assert(int8_cached_scaled_route(2048,7168,2048,8,true,80)==22);
 assert(int8_cached_scaled_route(8192,4096,4096,8,true,80)==23);
 assert(int8_cached_scaled_route(4096,5120,2304,48,true,80)==23);
 assert(int8_cached_scaled_route(2048,3584,3072,8,true,80)==23);
 assert(int8_cached_scaled_route(2048,8192,2048,32,true,80)==23);
 assert(int8_cached_scaled_route(2048,4096,1152,8,true,80)==22);
 // Keep partial/unprepared and uncalibrated ranges on their original policy.
 for(const auto shape : {std::array<int,3>{1024,4096,4096},{2176,4096,4096},
                         {2048,3328,4096},{2048,8448,4096},{2048,4096,896},
                         {2048,4096,16384},{2048,4096,1040},{2048,4160,4096},
                         {2048,2048,4096},{2048,6144,2048}})
  assert(int8_cached_scaled_route(shape[0],shape[1],shape[2],8,true,80)==
         int8_cached_route(shape[0],shape[1],shape[2],8,true,80));
 assert(int8_cached_scaled_route(8192,4096,4096,8,false,80)==int8_cached_route(8192,4096,4096,8,false,80));
 assert(int8_cached_scaled_route(8192,4096,4096,8,true,96)==int8_cached_route(8192,4096,4096,8,true,96));
 assert(int8_cached_route(8192,4096,4096,8,true,80)==12);
 // Scaled-only full-N threshold; native/device/pack boundaries remain explicit.
 assert(int8_cached_scaled_route(2048,4608,5120,12,true,80)==12);
 assert(int8_cached_route(2048,4608,5120,12,true,80)==5);
 assert(int8_cached_scaled_route(2048,4608,5120,12,false,80)==int8_cached_route(2048,4608,5120,12,false,80));
 assert(int8_cached_scaled_route(2048,4608,5120,12,true,96)==int8_cached_route(2048,4608,5120,12,true,96));
 assert(int8_cached_scaled_route(1024,4608,5120,12,true,80)==int8_cached_route(1024,4608,5120,12,true,80));
 assert(int8_cached_scaled_route(2176,4608,5120,12,true,80)==int8_cached_route(2176,4608,5120,12,true,80));
 using ck_tile::int8_cached_scaled_small_route;
 for(int m:{1,2,4,8})for(int k:{6144,7168})assert(int8_cached_scaled_small_route(m,4096,k,8,true,80)==24);
 assert(int8_cached_scaled_small_route(0,4096,6144,8,true,80)==0);
 assert(int8_cached_scaled_small_route(16,4096,6144,8,true,80)==0);
 assert(int8_cached_scaled_small_route(1,4096,6144,16,true,80)==0);
 for(int m:{1,2})for(int k:{7168,8192})assert(int8_cached_scaled_small_route(m,4096,k,16,false,80)==25);
 assert(int8_cached_scaled_small_route(3,4096,8192,64,true,80)==0);
 assert(int8_cached_scaled_small_route(1,4096,6144,8,false,80)==0);
 assert(int8_cached_scaled_small_route(1,4096,6144,8,true,96)==0);
 assert(int8_cached_scaled_small_route(1,4096,8192,8,true,80)==0);
 std::cout<<"SCALED_TAIL_POLICY_PASS\n";
}
