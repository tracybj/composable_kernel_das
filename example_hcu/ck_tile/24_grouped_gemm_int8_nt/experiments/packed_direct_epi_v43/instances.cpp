// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
template<class P> int run_direct(const void* args,int g,int m,int n,hipStream_t s){
 dim3 grid((n+P::BN-1)/P::BN,(m+P::BM-1)/P::BM,g);
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtPackedDirectEpiKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)
{
 if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
 auto s=reinterpret_cast<hipStream_t>(stream);
 switch(id){
 case 8701:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<32,64,128,1,2,false>>(args,g,m,n,s);
 case 8702:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<32,64,128,1,2,true>>(args,g,m,n,s);
 case 8703:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<32,64,256,1,2,false>>(args,g,m,n,s);
 case 8704:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<32,64,256,1,2,true>>(args,g,m,n,s);
 case 8705:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<32,128,256,2,2,true>>(args,g,m,n,s);
 case 8711:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<64,128,128,2,2,false>>(args,g,m,n,s);
 case 8712:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<64,128,128,2,2,true>>(args,g,m,n,s);
 case 8713:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<64,128,256,2,2,false>>(args,g,m,n,s);
 case 8714:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<64,128,256,2,2,true>>(args,g,m,n,s);
 case 8721:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<128,128,64,2,2,false>>(args,g,m,n,s);
 case 8722:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<128,128,64,2,2,true>>(args,g,m,n,s);
 case 8723:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<128,64,128,2,2,true>>(args,g,m,n,s);
 case 8731:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<16,64,128,1,2,true>>(args,g,m,n,s);
 case 8732:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<16,64,256,1,2,true>>(args,g,m,n,s);
 case 8733:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<16,128,256,1,4,true>>(args,g,m,n,s);
 case 8734:return run_direct<ck_tile::GroupedInt8NtPackedDirectEpiProblem<16,32,512,1,2,true>>(args,g,m,n,s);
 default:return baseline_launch(id,args,g,m,n,stream);
 }
}
