// SPDX-License-Identifier: MIT
#include "../../cached_weights/scaled_kernel.hpp"
#include "../../cached_weights/core_kernel.hpp"
#include "../../cached_weights/instances.hpp"
#include "../packed_output_v129/epilogue.hpp"
#include "../packed_exchange_v141/epilogue.hpp"
#include "../finite_factor_v142/epilogue.hpp"
namespace {
template<class P,bool PackedOutput=false,bool Packed16=false,bool FiniteFactor=false>int run(unsigned output,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
    const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
    const auto* a=static_cast<const ck_tile::GemmTransKernelArg*>(args);const auto* s=static_cast<const ck_tile::Int8NtScaleArgs*>(scales);
    const ck_tile::stream_config config{stream,false};
    if(output==0)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspCoreKernel<P>{},grid,dim3(P::Threads),0,a));
    else if(output==1)ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtScaledWaspKernel<P,true,std::conditional_t<FiniteFactor,ck_tile::GroupedInt8NtFiniteFactorEpilogue<P,true>,std::conditional_t<Packed16,ck_tile::GroupedInt8NtPacked16ExchangeEpilogue<P,true>,std::conditional_t<PackedOutput,ck_tile::GroupedInt8NtPackedOutputEpilogue<P,true>,ck_tile::GroupedInt8NtScaledWaspEpilogue<P,true>>>>>{},grid,dim3(P::Threads),0,a,s));
    else ck_tile::launch_kernel(config,ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtScaledWaspKernel<P,false,std::conditional_t<FiniteFactor,ck_tile::GroupedInt8NtFiniteFactorEpilogue<P,false>,std::conditional_t<Packed16,ck_tile::GroupedInt8NtPacked16ExchangeEpilogue<P,false>,std::conditional_t<PackedOutput,ck_tile::GroupedInt8NtPackedOutputEpilogue<P,false>,ck_tile::GroupedInt8NtScaledWaspEpilogue<P,false>>>>>{},grid,dim3(P::Threads),0,a,s));
    return hipGetLastError()==hipSuccess?0:-100;
}
template<int N,int Batch,int Group>using G256=ck_tile::GroupedInt8NtWaspCoreProblem<256,false,N/32,Group,Batch,true,true,0,false,true,0,N/32>;

template<int Group>struct TailMaskedG256 : G256<384,2,Group> {
    using Base=G256<384,2,Group>;
    using Pipeline=ck_tile::GroupedInt8NtWaspPipeline<typename Base::PipelineProblem,
        ck_tile::GemmPipelineAgBgCrCompV4DefaultPolicy,12,2,true,true,0,false,true,0,true>;
    static_assert(Pipeline::BlockSize==Base::Threads && Pipeline::kWaspLdsBytes==65536);
};
template<int N,int Batch,int Group>using P=ck_tile::GroupedInt8NtWaspCoreProblem<128,false,N/32,Group,Batch,true,true,0,false,true,0,N/32>;
template<int Batch,int Group>struct SingleB128 : P<384,Batch,Group> {
    using Base=P<384,Batch,Group>;
    using Pipeline=ck_tile::GroupedInt8NtWaspPipeline<typename Base::PipelineProblem,
        ck_tile::GemmPipelineAgBgCrCompV4DefaultPolicy,12,Batch,true,true,0,false,true,0,false,true>;
    static_assert(Pipeline::BlockSize==896 && Pipeline::kWaspLdsBytes==32768);
};
template<int Consumers,int Batch,int Group,bool Mask>struct CrossHalf256 : G256<Consumers*32,Batch,Group> {
    using Base=G256<Consumers*32,Batch,Group>;
    using Pipeline=ck_tile::GroupedInt8NtWaspPipeline<typename Base::PipelineProblem,
        ck_tile::GemmPipelineAgBgCrCompV4DefaultPolicy,Consumers,Batch,true,true,0,false,true,0,Mask,false,true>;
    static_assert(Pipeline::BlockSize==Base::Threads && Pipeline::kWaspLdsBytes==65536);
};
}
extern "C" int ck_tile_int8_experimental_wasp_launch_private(unsigned output,unsigned route,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
    switch(route){
    case 13:return run<P<384,2,4>>(output,args,scales,g,m,n,stream);
    case 14:return run<P<384,4,4>>(output,args,scales,g,m,n,stream);
    case 15:return run<P<256,4,4>>(output,args,scales,g,m,n,stream);
    case 16:return run<P<384,2,8>>(output,args,scales,g,m,n,stream);
    case 17:return run<P<384,4,8>>(output,args,scales,g,m,n,stream);
    case 18:return run<ck_tile::int8_nt_instances::PackedM256N384,true>(output,args,scales,g,m,n,stream);
    case 19:return run<ck_tile::int8_nt_instances::PackedM256N256,true>(output,args,scales,g,m,n,stream);
    case 20:return run<ck_tile::int8_nt_instances::PackedM128N256,true>(output,args,scales,g,m,n,stream);
    case 21:return run<ck_tile::int8_nt_instances::PackedM128N128,true>(output,args,scales,g,m,n,stream);
    case 48:return run<G256<384,2,1>>(output,args,scales,g,m,n,stream);
    case 49:return run<G256<384,2,2>>(output,args,scales,g,m,n,stream);
    case 50:return run<G256<384,2,8>>(output,args,scales,g,m,n,stream);
    case 51:return run<G256<384,2,16>>(output,args,scales,g,m,n,stream);
    case 52:return run<G256<256,4,1>>(output,args,scales,g,m,n,stream);
    case 53:return run<G256<256,4,2>>(output,args,scales,g,m,n,stream);
    case 54:return run<G256<256,4,8>>(output,args,scales,g,m,n,stream);
    case 55:return run<G256<256,4,16>>(output,args,scales,g,m,n,stream);
    case 56:return run<TailMaskedG256<1>>(output,args,scales,g,m,n,stream);
    case 57:return run<TailMaskedG256<2>>(output,args,scales,g,m,n,stream);
    case 58:return run<TailMaskedG256<4>>(output,args,scales,g,m,n,stream);
    case 59:return run<TailMaskedG256<8>>(output,args,scales,g,m,n,stream);
    case 60:return run<TailMaskedG256<16>>(output,args,scales,g,m,n,stream);
    case 61:return run<SingleB128<1,2>>(output,args,scales,g,m,n,stream);
    case 62:return run<SingleB128<1,4>>(output,args,scales,g,m,n,stream);
    case 63:return run<SingleB128<2,2>>(output,args,scales,g,m,n,stream);
    case 64:return run<SingleB128<2,4>>(output,args,scales,g,m,n,stream);
    case 65:return run<P<384,1,2>>(output,args,scales,g,m,n,stream);
    case 66:return run<P<384,1,4>>(output,args,scales,g,m,n,stream);
    case 67:return run<P<384,2,2>>(output,args,scales,g,m,n,stream);
    case 68:return run<ck_tile::int8_nt_instances::PackedM256N384,false,true>(output,args,scales,g,m,n,stream);
    case 69:return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM4,false,true>(output,args,scales,g,m,n,stream);
    case 70:return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM8,false,true>(output,args,scales,g,m,n,stream);
    case 71:return run<ck_tile::int8_nt_instances::PackedM256N256,false,true>(output,args,scales,g,m,n,stream);
    case 72:return run<ck_tile::int8_nt_instances::PackedM256N384,false,false,true>(output,args,scales,g,m,n,stream);
    case 73:return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM4,false,false,true>(output,args,scales,g,m,n,stream);
    case 74:return run<ck_tile::int8_nt_instances::PackedM256N384MaskedM8,false,false,true>(output,args,scales,g,m,n,stream);
    case 75:return run<ck_tile::int8_nt_instances::PackedM256N256,false,false,true>(output,args,scales,g,m,n,stream);
    case 76:return run<CrossHalf256<12,2,4,false>>(output,args,scales,g,m,n,stream);
    case 77:return run<CrossHalf256<12,2,4,true>>(output,args,scales,g,m,n,stream);
    case 78:return run<CrossHalf256<12,2,8,true>>(output,args,scales,g,m,n,stream);
    case 79:return run<CrossHalf256<8,4,4,false>>(output,args,scales,g,m,n,stream);
    default:return -5;
    }
}
