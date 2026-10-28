// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../direct_lds_v28/kernel.hpp"
#include "kernel.hpp"
namespace {
template<class P,template<class> class Kernel>
int run(const void* args,int g,int m,int n,hipStream_t stream)
{
 const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(Kernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)
{
 if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
 auto s=reinterpret_cast<hipStream_t>(stream);
 switch(id){
 case 6702:return run<ck_tile::GroupedInt8NtD2LProblem<true,false>,ck_tile::GroupedInt8NtD2LKernel>(args,g,m,n,s);
 case 7101:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<4,2,1>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7104:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<4,2,4>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7108:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<4,2,8>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7116:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<4,2,16>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7201:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,4,1>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7204:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,4,4>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7208:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,4,8>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7216:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,4,16>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7301:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,2,1>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 7401:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<4,4,1>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 default:return baseline_launch(id,args,g,m,n,stream);
 }
}
