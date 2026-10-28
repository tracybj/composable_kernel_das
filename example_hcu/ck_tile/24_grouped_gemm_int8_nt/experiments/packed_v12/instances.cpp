// SPDX-License-Identifier: MIT

#include "grouped_int8_nt_tile.hpp"

#include "grouped_int8_nt_direct.hpp"

#include <hip/hip_runtime.h>



namespace {

template<class P>

int launch(const void* args,int groups,int m,int n,hipStream_t stream)

{

    const dim3 grid(P::MFirst?(m+P::BM-1)/P::BM:(n+P::BN-1)/P::BN,

                    P::MFirst?(n+P::BN-1)/P::BN:(m+P::BM-1)/P::BM,groups);

    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},

        ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtTileKernel<P>{},grid,dim3(P::Threads),0,

            static_cast<const ck_tile::GemmTransKernelArg*>(args)));

    return hipGetLastError()==hipSuccess?0:-100;

}

}

extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)

{

    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;

    auto s=reinterpret_cast<hipStream_t>(stream);

    switch(id){

    case 6004:{

        const dim3 grid((n+63)/64,(m+15)/16,g);

        ck_tile::launch_kernel(ck_tile::stream_config{s,false},

            ck_tile::make_kernel<256,1>(ck_tile::GroupedInt8NtDirectKernel<>{},grid,dim3(256),0,

                static_cast<const ck_tile::GemmTransKernelArg*>(args)));

        return hipGetLastError()==hipSuccess?0:-100;

    }

    case 6005:return launch<ck_tile::GroupedInt8NtTileProblem<32,128>>(args,g,m,n,s);

    case 6006:return launch<ck_tile::GroupedInt8NtTileProblem<64,128>>(args,g,m,n,s);

    case 6007:return launch<ck_tile::GroupedInt8NtTileProblem<128,128>>(args,g,m,n,s);

    case 6008:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4>>(args,g,m,n,s);

    case 6010:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2>>(args,g,m,n,s);

    case 6011:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2>>(args,g,m,n,s);

    case 6012:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,2>>(args,g,m,n,s);

    case 6013:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true>>(args,g,m,n,s);

    case 6014:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,32,4,2,true>>(args,g,m,n,s);

    case 6015:return launch<ck_tile::GroupedInt8NtTileProblem<512,128,64,8,1,true>>(args,g,m,n,s);

    case 6016:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true>>(args,g,m,n,s);

    case 6017:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,true>>(args,g,m,n,s);

    case 6018:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,128,4,2,true>>(args,g,m,n,s);

    case 6019:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,true>>(args,g,m,n,s);

    case 6020:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false>>(args,g,m,n,s);

    case 6021:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,true>>(args,g,m,n,s);

    case 6022:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,false,true,false>>(args,g,m,n,s);

    case 6023:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4,true,false,true,true>>(args,g,m,n,s);

    case 6024:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,false>>(args,g,m,n,s);

    case 6025:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6026:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6027:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,true,true,true>>(args,g,m,n,s);

    case 6028:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6029:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,2,4,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6030:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,4,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6031:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,0>>(args,g,m,n,s);

    case 6032:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,32>>(args,g,m,n,s);

    case 6033:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,false,true,false,true,true>>(args,g,m,n,s);

    case 6034:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,16,true>>(args,g,m,n,s);

    case 6035:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,false,true,false,true,true,16,true>>(args,g,m,n,s);

    case 6036:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4,true,false,true,false,true,true,16,true>>(args,g,m,n,s);

    case 6037:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,4,true,false,true,false,true,true,16,true>>(args,g,m,n,s);

    case 6038:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,16,false,2,false>>(args,g,m,n,s);

    case 6039:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,16,false,1,true>>(args,g,m,n,s);

    case 6040:return launch<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,false,true,false,true,true,16,false,2,true>>(args,g,m,n,s);

    case 6041:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,false,true,false,true,true,16,false,1,true>>(args,g,m,n,s);

    case 6042:return launch<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4,true,false,true,false,true,true,16,false,2,true>>(args,g,m,n,s);

    default:return -5;

    }

}
