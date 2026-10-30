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
 std::cout<<"WAVE_AND_M32_POLICY_PASS checks=11\n";
}
