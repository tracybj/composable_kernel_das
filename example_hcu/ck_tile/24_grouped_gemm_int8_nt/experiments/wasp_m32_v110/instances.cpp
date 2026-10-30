// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run(const void* args,int g,int m,int n,hipStream_t stream){
 const int mt=(m+31)/32,nt=(n+P::BN-1)/P::BN;const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspM32Kernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,hipStream_t stream){switch(id){
case 41001:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,8,1,2,true,true,0,false,true,1,8>>(args,g,m,n,stream);
case 41002:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,8,1,1,true,true,0,false,true,1,8>>(args,g,m,n,stream);
case 41101:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,4,1,2,true,true,0,false,true,1,4>>(args,g,m,n,stream);
case 41102:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,4,1,1,true,true,0,false,true,1,4>>(args,g,m,n,stream);
case 41201:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,2,1,2,true,true,0,false,true,1,2>>(args,g,m,n,stream);
case 41202:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,2,1,1,true,true,0,false,true,1,2>>(args,g,m,n,stream);
case 41011:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,8,1,2,true,true,0,false,false,1,8>>(args,g,m,n,stream);
case 41211:return run<ck_tile::GroupedInt8NtWaspM32Problem<false,2,1,2,true,true,0,false,false,1,2>>(args,g,m,n,stream);
default:return baseline_launch(id,args,g,m,n,stream);}}
