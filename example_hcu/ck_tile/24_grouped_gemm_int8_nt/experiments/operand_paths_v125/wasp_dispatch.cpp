// SPDX-License-Identifier: MIT
#include "../../cached_weights/scaled_kernel.hpp"
#include "../../cached_weights/core_kernel.hpp"
namespace {
template<class P>int run(unsigned output,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
    const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
    const auto* a=static_cast<const ck_tile::GemmTransKernelArg*>(args);const auto* s=static_cast<const ck_tile::Int8NtScaleArgs*>(scales);
    const ck_tile::stream_config config{stream,false};
    if(output==0)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCoreKernel<P>{},grid,dim3(P::Threads),0,a));
    else if(output==1)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtScaledWaspKernel<P,true>{},grid,dim3(P::Threads),0,a,s));
    else ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtScaledWaspKernel<P,false>{},grid,dim3(P::Threads),0,a,s));
    return hipGetLastError()==hipSuccess?0:-100;
}
template<int N,int Batch,int Group>using P=ck_tile::GroupedInt8NtWaspCoreProblem<128,false,N/32,Group,Batch,true,true,0,false,true,0,N/32>;
}
extern "C" int ck_tile_int8_wide128_launch_private(unsigned output,unsigned route,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    switch(route){
    case 13:return run<P<384,2,4>>(output,args,scales,g,m,n,stream);
    case 14:return run<P<384,4,4>>(output,args,scales,g,m,n,stream);
    case 15:return run<P<256,4,4>>(output,args,scales,g,m,n,stream);
    case 16:return run<P<384,2,8>>(output,args,scales,g,m,n,stream);
    case 17:return run<P<384,4,8>>(output,args,scales,g,m,n,stream);
    default:return -5;
    }
}
