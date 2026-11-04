// SPDX-License-Identifier: MIT
#include "direct_pipeline.hpp"
#include "../../cached_weights/scaled_kernel.hpp"
#include "../../grouped_int8_nt_kernel.hpp"
namespace {
template<class P>int run(unsigned output,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    dim3 grid(P::MFirst?(m+P::BM-1)/P::BM:(n+P::BN-1)/P::BN,P::MFirst?(n+P::BN-1)/P::BN:(m+P::BM-1)/P::BM,g);
    if constexpr(P::MGroup>0){const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;grid=dim3(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);}
    const auto* a=static_cast<const ck_tile::GemmTransKernelArg*>(args);const auto* s=static_cast<const ck_tile::Int8NtScaleArgs*>(scales);
    const ck_tile::stream_config config{stream,false};
    if(output==0)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtTileKernel<P>{},grid,dim3(P::Threads),0,a));
    else if(output==1)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtScaledTileKernel<P,true>{},grid,dim3(P::Threads),0,a,s));
    else ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtScaledTileKernel<P,false>{},grid,dim3(P::Threads),0,a,s));
    return hipGetLastError()==hipSuccess?0:-100;
}
using B128=ck_tile::GroupedInt8NtTileProblem<16,64,128,1,2,false,false,true,true,1,true>;
using B256=ck_tile::GroupedInt8NtTileProblem<16,64,256,1,2,false,false,true,true,1,true>;
using N128=ck_tile::GroupedInt8NtTileProblem<16,128,256,1,4,false,false,true,true,1,true>;
using M128=ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,false,true,false,1,true,8>;
template<class Base,bool A,bool B>using Path=ck_tile::GroupedInt8NtOperandProblem<Base,A,B>;
}
extern "C" int ck_tile_int8_operand_launch_private(unsigned output,unsigned id,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    switch(id){
    case 9001:return run<Path<B128,true,false>>(output,args,scales,g,m,n,stream);
    case 9002:return run<Path<B128,false,true>>(output,args,scales,g,m,n,stream);
    case 9003:return run<Path<B128,false,false>>(output,args,scales,g,m,n,stream);
    case 9004:return run<Path<B256,true,false>>(output,args,scales,g,m,n,stream);
    case 9005:return run<Path<B256,false,true>>(output,args,scales,g,m,n,stream);
    case 9006:return run<Path<B256,false,false>>(output,args,scales,g,m,n,stream);
    case 9007:return run<Path<N128,true,false>>(output,args,scales,g,m,n,stream);
    case 9008:return run<Path<N128,false,true>>(output,args,scales,g,m,n,stream);
    case 9009:return run<Path<N128,false,false>>(output,args,scales,g,m,n,stream);
    case 9010:return run<Path<M128,true,false>>(output,args,scales,g,m,n,stream);
    case 9011:return run<Path<M128,false,true>>(output,args,scales,g,m,n,stream);
    case 9012:return run<Path<M128,false,false>>(output,args,scales,g,m,n,stream);
    case 9013:return run<ck_tile::GroupedInt8NtOperandProblem<B128,true,false,true>>(output,args,scales,g,m,n,stream);
    case 9014:return run<ck_tile::GroupedInt8NtOperandProblem<B128,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9015:return run<ck_tile::GroupedInt8NtOperandProblem<B128,false,false,true>>(output,args,scales,g,m,n,stream);
    case 9016:return run<ck_tile::GroupedInt8NtOperandProblem<B256,true,false,true>>(output,args,scales,g,m,n,stream);
    case 9017:return run<ck_tile::GroupedInt8NtOperandProblem<B256,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9018:return run<ck_tile::GroupedInt8NtOperandProblem<B256,false,false,true>>(output,args,scales,g,m,n,stream);
    case 9019:return run<ck_tile::GroupedInt8NtOperandProblem<N128,true,false,true>>(output,args,scales,g,m,n,stream);
    case 9020:return run<ck_tile::GroupedInt8NtOperandProblem<N128,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9021:return run<ck_tile::GroupedInt8NtOperandProblem<N128,false,false,true>>(output,args,scales,g,m,n,stream);
    case 9022:return run<ck_tile::GroupedInt8NtOperandProblem<B128,true,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9023:return run<ck_tile::GroupedInt8NtOperandProblem<B256,true,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9024:return run<ck_tile::GroupedInt8NtOperandProblem<N128,true,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9031:return run<ck_tile::GroupedInt8NtOperandProblem<ck_tile::GroupedInt8NtTileProblem<16,128,128,1,4,false,false,true,true,1,true>,true,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9032:return run<ck_tile::GroupedInt8NtOperandProblem<ck_tile::GroupedInt8NtTileProblem<16,64,64,1,2,false,false,true,true,1,true>,true,false,true,true>>(output,args,scales,g,m,n,stream);
    case 9034:return run<Path<B128,true,true>>(output,args,scales,g,m,n,stream);
    case 9035:return run<Path<B256,true,true>>(output,args,scales,g,m,n,stream);
    case 9036:return run<Path<N128,true,true>>(output,args,scales,g,m,n,stream);
    default:return -5;
    }
}
