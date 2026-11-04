#include "selector.hpp"
#include <cassert>
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
 std::cout<<"WAVE_AND_M32_POLICY_PASS checks=24\n";
}
