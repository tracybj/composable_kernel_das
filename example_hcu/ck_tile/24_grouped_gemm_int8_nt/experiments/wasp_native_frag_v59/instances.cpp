// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../direct_lds_v28/kernel.hpp"
#include "../aligned_d2l_v39/kernel.hpp"
#include "../direct_epilogue_v42/kernel.hpp"
#include "../dense_k128_v46/kernel.hpp"
#include "../split_roles_v48/kernel.hpp"
#include "../bypass_b_v50/kernel.hpp"
#include "../bypass_b_ordered_v52/kernel.hpp"
#include "../transposed_bypass_v54/kernel.hpp"
#include "../hoisted_copy_v55/kernel.hpp"
#include "../wasp_struct_b_v57/kernel.hpp"
#include "../wasp_int8_port_v58/kernel.hpp"
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
namespace {
template<class P,template<class> class Kernel=ck_tile::GroupedInt8NtTransposedKernel> int run_transposed(const void* args,int g,int m,int n,hipStream_t stream)
{
 const int mt=(n+P::BM-1)/P::BM,nt=(m+P::BN-1)/P::BN;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(Kernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
namespace {
template<class P> int run_struct_b(const void* args,int g,int m,int n,hipStream_t stream)
{
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n;
 const int mt=(mm+P::BM-1)/P::BM,nt=(nn+P::BN-1)/P::BN;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtStructBKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
namespace {
template<class P> int run_wasp_port(const void* args,int g,int m,int n,hipStream_t stream)
{
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n;
 const int mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspPortKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
namespace {
template<class P> int run_wasp_native(const void* args,int g,int m,int n,hipStream_t stream)
{
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n;
 const int mt=(mm+255)/256,nt=(nn+255)/256;
 dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspNativeKernel<P>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
 return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)
{
 if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
 auto s=reinterpret_cast<hipStream_t>(stream);
 switch(id){
 case 7204:return run<ck_tile::GroupedInt8NtD2LGeometryProblem<2,4,4>,ck_tile::GroupedInt8NtD2LGeometryKernel>(args,g,m,n,s);
 case 8402:return run<ck_tile::GroupedInt8NtD2LAlignedProblem<2,4,4,true>,ck_tile::GroupedInt8NtD2LAlignedKernel>(args,g,m,n,s);
 case 8602:return run<ck_tile::GroupedInt8NtDirectEpiProblem<2,4,4,true>,ck_tile::GroupedInt8NtDirectEpiKernel>(args,g,m,n,s);
 case 8901:return run<ck_tile::GroupedInt8NtDenseKProblem<0,false,true,true,4,2,4,true>,ck_tile::GroupedInt8NtDenseKKernel>(args,g,m,n,s);
 case 8902:return run<ck_tile::GroupedInt8NtDenseKProblem<4,false,true,true,4,2,4,true>,ck_tile::GroupedInt8NtDenseKKernel>(args,g,m,n,s);
 case 9001:return run<ck_tile::GroupedInt8NtSplitRolesProblem<false,true,true,4,2,4,false>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9002:return run<ck_tile::GroupedInt8NtSplitRolesProblem<false,true,true,4,2,8,false>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9003:return run<ck_tile::GroupedInt8NtSplitRolesProblem<false,true,true,2,4,4,false>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9004:return run<ck_tile::GroupedInt8NtSplitRolesProblem<true,false,true,4,2,4,false>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9005:return run<ck_tile::GroupedInt8NtSplitRolesProblem<false,true,false,4,2,4,false>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9006:return run<ck_tile::GroupedInt8NtSplitRolesProblem<false,true,true,4,2,4,true>,ck_tile::GroupedInt8NtSplitRolesKernel>(args,g,m,n,s);
 case 9101:return run<ck_tile::GroupedInt8NtBypassBProblem<true,true,4,false,0>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9102:return run<ck_tile::GroupedInt8NtBypassBProblem<true,true,8,false,0>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9103:return run<ck_tile::GroupedInt8NtBypassBProblem<true,false,4,false,0>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9104:return run<ck_tile::GroupedInt8NtBypassBProblem<false,true,4,false,0>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9105:return run<ck_tile::GroupedInt8NtBypassBProblem<true,true,4,true,0>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9106:return run<ck_tile::GroupedInt8NtBypassBProblem<true,true,4,false,1>,ck_tile::GroupedInt8NtBypassBKernel>(args,g,m,n,s);
 case 9301:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<true,true,4,false,0,2>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9302:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<true,true,4,false,0,3>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9303:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<true,false,4,false,0,2>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9304:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<false,true,4,false,0,2>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9305:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<true,true,4,true,0,2>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9306:return run<ck_tile::GroupedInt8NtBypassBOrderedProblem<true,true,8,false,0,3>,ck_tile::GroupedInt8NtBypassBOrderedKernel>(args,g,m,n,s);
 case 9401:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<true,true,4,false,2>>(args,g,m,n,s);
 case 9402:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<true,true,4,false,3>>(args,g,m,n,s);
 case 9403:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<true,true,8,false,2>>(args,g,m,n,s);
 case 9404:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<false,true,4,false,2>>(args,g,m,n,s);
 case 9405:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<true,true,4,true,2>>(args,g,m,n,s);
 case 9406:return run_transposed<ck_tile::GroupedInt8NtTransposedProblem<true,false,4,false,2>>(args,g,m,n,s);
 case 9501:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<true,4,false,true>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9502:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<true,8,false,true>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9503:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<true,4,true,true>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9504:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<false,4,false,true>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9505:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<true,4,false,false>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9506:return run_transposed<ck_tile::GroupedInt8NtHoistedCopyProblem<false,4,false,false>,ck_tile::GroupedInt8NtHoistedTransposedKernel>(args,g,m,n,s);
 case 9507:return run<ck_tile::GroupedInt8NtHoistedCopyProblem<true,4,false,true>,ck_tile::GroupedInt8NtHoistedNativeKernel>(args,g,m,n,s);
 case 9508:return run<ck_tile::GroupedInt8NtHoistedCopyProblem<false,4,false,true>,ck_tile::GroupedInt8NtHoistedNativeKernel>(args,g,m,n,s);
 case 9801:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<true,true,4,false>>(args,g,m,n,s);
 case 9802:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<true,false,4,false>>(args,g,m,n,s);
 case 9803:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<false,true,4,false>>(args,g,m,n,s);
 case 9804:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<false,false,4,false>>(args,g,m,n,s);
 case 9805:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<true,true,4,true>>(args,g,m,n,s);
 case 9806:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<false,true,4,true>>(args,g,m,n,s);
 case 9807:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<true,true,8,false>>(args,g,m,n,s);
 case 9808:return run_struct_b<ck_tile::GroupedInt8NtStructBProblem<false,true,8,false>>(args,g,m,n,s);
 case 9901:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<false,8,4>>(args,g,m,n,s);
 case 9902:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<false,0,4>>(args,g,m,n,s);
 case 9903:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<false,4,4>>(args,g,m,n,s);
 case 9904:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<true,8,4>>(args,g,m,n,s);
 case 9905:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<true,0,4>>(args,g,m,n,s);
 case 9906:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<true,4,4>>(args,g,m,n,s);
 case 9907:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<false,8,8>>(args,g,m,n,s);
 case 9908:return run_wasp_port<ck_tile::GroupedInt8NtWaspPortProblem<true,8,8>>(args,g,m,n,s);
 case 10001:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<false,8,4>>(args,g,m,n,s);
 case 10002:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<false,0,4>>(args,g,m,n,s);
 case 10003:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<false,4,4>>(args,g,m,n,s);
 case 10004:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<true,8,4>>(args,g,m,n,s);
 case 10005:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<true,0,4>>(args,g,m,n,s);
 case 10006:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<true,4,4>>(args,g,m,n,s);
 case 10007:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<false,8,8>>(args,g,m,n,s);
 case 10008:return run_wasp_native<ck_tile::GroupedInt8NtWaspNativeProblem<true,8,8>>(args,g,m,n,s);
 default:return baseline_launch(id,args,g,m,n,stream);
 }
}
