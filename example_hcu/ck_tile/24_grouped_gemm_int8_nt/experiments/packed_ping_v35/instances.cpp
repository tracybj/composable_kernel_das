// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run_ping(const void* args,int g,int m,int n,hipStream_t s)
{
 dim3 grid((n+P::BN-1)/P::BN,(m+P::BM-1)/P::BM,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtPackedPingKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)
{
 if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
 auto s=reinterpret_cast<hipStream_t>(stream);
 switch(id){
 case 7501:return launch<ck_tile::GroupedInt8NtTileProblem<32,64,128,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7502:return launch<ck_tile::GroupedInt8NtTileProblem<32,64,256,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7503:return launch<ck_tile::GroupedInt8NtTileProblem<32,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7504:return launch<ck_tile::GroupedInt8NtTileProblem<32,128,256,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7505:return launch<ck_tile::GroupedInt8NtTileProblem<32,256,128,2,4,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7601:return launch<ck_tile::GroupedInt8NtTileProblem<64,64,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7602:return launch<ck_tile::GroupedInt8NtTileProblem<64,64,256,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7603:return launch<ck_tile::GroupedInt8NtTileProblem<64,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7604:return launch<ck_tile::GroupedInt8NtTileProblem<64,128,256,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7605:return launch<ck_tile::GroupedInt8NtTileProblem<64,256,128,2,4,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7703:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7901:return launch<ck_tile::GroupedInt8NtTileProblem<32,32,512,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7902:return launch<ck_tile::GroupedInt8NtTileProblem<32,64,512,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7903:return launch<ck_tile::GroupedInt8NtTileProblem<32,64,256,2,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7951:return launch<ck_tile::GroupedInt8NtTileProblem<64,32,512,2,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7952:return launch<ck_tile::GroupedInt8NtTileProblem<64,64,256,2,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7953:return launch<ck_tile::GroupedInt8NtTileProblem<64,128,128,2,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7954:return launch<ck_tile::GroupedInt8NtTileProblem<64,128,256,2,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7955:return launch<ck_tile::GroupedInt8NtTileProblem<64,128,256,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7956:return launch<ck_tile::GroupedInt8NtTileProblem<64,256,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8201:return run_ping<ck_tile::GroupedInt8NtTileProblem<32,64,64,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8202:return run_ping<ck_tile::GroupedInt8NtTileProblem<32,64,128,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8203:return run_ping<ck_tile::GroupedInt8NtTileProblem<32,128,64,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8204:return run_ping<ck_tile::GroupedInt8NtTileProblem<32,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8301:return run_ping<ck_tile::GroupedInt8NtTileProblem<64,64,64,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8302:return run_ping<ck_tile::GroupedInt8NtTileProblem<64,128,64,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8303:return run_ping<ck_tile::GroupedInt8NtTileProblem<64,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8304:return run_ping<ck_tile::GroupedInt8NtTileProblem<64,256,64,2,4,false,false,true,true,1,true>>(args,g,m,n,s);
 default:return baseline_launch(id,args,g,m,n,stream);
 }
}
