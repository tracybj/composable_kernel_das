// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "kernel.hpp"
namespace {
enum class DirectLdsInstance:unsigned {single=6701,ping_pong=6702,merged=6703,
    producers4=6704,producers4_mgroup8=6705};
template<class P>
int launch_direct(const void* args,int g,int m,int n,hipStream_t stream)
{
    const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
    dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtD2LKernel<P>{},grid,
            dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
    auto s=reinterpret_cast<hipStream_t>(stream);
    switch(static_cast<DirectLdsInstance>(instance_id))
    {
        case DirectLdsInstance::single:return launch_direct<ck_tile::GroupedInt8NtD2LProblem<false,false>>(args,g,m,n,s);
        case DirectLdsInstance::ping_pong:return launch_direct<ck_tile::GroupedInt8NtD2LProblem<true,false>>(args,g,m,n,s);
        case DirectLdsInstance::merged:return launch_direct<ck_tile::GroupedInt8NtD2LProblem<true,true>>(args,g,m,n,s);
        case DirectLdsInstance::producers4:return launch_direct<ck_tile::GroupedInt8NtD2LProblem<true,true,4>>(args,g,m,n,s);
        case DirectLdsInstance::producers4_mgroup8:return launch_direct<ck_tile::GroupedInt8NtD2LProblem<true,true,4,8>>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
