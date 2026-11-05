// SPDX-License-Identifier: MIT
#include "scaled_kernel.hpp"
#include "instances.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#ifdef CK_INT8_CACHED_TUNING
extern "C" int ck_tile_int8_operand_launch_private(unsigned,unsigned,const void*,const void*,int,int,int,hipStream_t);
extern "C" int ck_tile_int8_packed_small_launch_private(unsigned,unsigned,const void*,const void*,int,int,int,hipStream_t);
extern "C" int ck_tile_int8_experimental_wasp_launch_private(unsigned,unsigned,const void*,const void*,int,int,int,hipStream_t);
#endif
namespace {
template<bool BF,class P> int wasp(const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
 const int mm=P::Transpose?n:m,nn=P::Transpose?m:n;
 const int mt=(mm+P::BM-1)/P::BM,nt=(nn+P::BN-1)/P::BN;
 const dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtScaledWaspKernel<P,BF>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args),static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
 return hipGetLastError()==hipSuccess?0:-100;
}
template<bool BF,int Batch,bool Transpose,int Cap=0> int legacy_wasp(const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
 using P=ck_tile::GroupedInt8NtWaspCoreProblem<256,Transpose,8,4,Batch,true,true,Cap>;
 return wasp<BF,P>(args,scales,g,m,n,stream);
}
template<bool BF,class P> int cooperative(const void* args,const void* scales,int groups,int m,int n,hipStream_t stream){
 dim3 grid(P::MFirst?(m+P::BM-1)/P::BM:(n+P::BN-1)/P::BN,P::MFirst?(n+P::BN-1)/P::BN:(m+P::BM-1)/P::BM,groups);
 if constexpr(P::MGroup>0){const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;grid=dim3(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,groups);}
 ck_tile::launch_kernel(ck_tile::stream_config{stream,false},ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtScaledTileKernel<P,BF>{},grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args),static_cast<const ck_tile::Int8NtScaleArgs*>(scales)));
 return hipGetLastError()==hipSuccess?0:-100;
}
template<bool BFloat> int raw(unsigned legacy,const void* args,const void* scales,int g,int m,int n,hipStream_t s){
 using Instance=ck_tile::GroupedGemmInstanceId;
 switch(static_cast<Instance>(legacy)){
    case Instance::int8_nt_m32: return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<32, 128>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m64: return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<64, 128>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m128: return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<128, 128>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m128_n256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<128, 256, 64, 2, 4>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m128_aligned:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m256_n256_aligned:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m128_n256_aligned:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<128, 256, 64, 2, 4, true, true, true, false, 2, true>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m256_n256_padded:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, false, true, false, 1, true>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m128_padded:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m256_n256_ngroup:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true, 1>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m128_mgroup8:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false, 8>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m128_padded_mgroup8:
        return cooperative<BFloat,
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true, 8>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_m16_n256_aligned:
        return cooperative<BFloat,
            ck_tile::GroupedInt8NtTileProblem<16, 256, 64, 1, 4, false, true, true, true, 1, true>>(
            args, scales, g, m, n, s);
    case Instance::int8_nt_small_m_n64_k128:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<16, 64, 128, 1, 2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_small_m_n64_k256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<16, 64, 256, 1, 2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_small_m_n128_k256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<16, 128, 256, 1, 4,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_small_g_n32_k512:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<16, 32, 512, 1, 2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m32_n64_k128:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<32,64,128,1,2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m32_n64_k256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<32,64,256,1,2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m32_n128_k256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<32,128,256,2,2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m64_n128_k128:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<64,128,128,2,2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_m64_n128_k256:
        return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<64,128,256,2,2,
            false, false, true, true, 1, true>>(args, scales, g, m, n, s);
    case Instance::int8_nt_small_m_n32_k512_direct: {
        using P = ck_tile::GroupedInt8NtTileProblem<16,32,512,1,2,
            false,false,true,true,1,true>;
        return cooperative<BFloat,P>(args,scales,g,m,n,s);
    }
    case Instance::int8_nt_wasp_a4_transposed: return legacy_wasp<BFloat,4,true>(args,scales,g,m,n,s);
    case Instance::int8_nt_wasp_a8_transposed: return legacy_wasp<BFloat,8,true>(args,scales,g,m,n,s);
    case Instance::int8_nt_wasp_a4: return legacy_wasp<BFloat,4,false>(args,scales,g,m,n,s);
    case Instance::int8_nt_wasp_record1024_a4: return legacy_wasp<BFloat,4,false,1024>(args,scales,g,m,n,s);
    case Instance::int8_nt_wasp_record1024_a4_transposed: return legacy_wasp<BFloat,4,true,1024>(args,scales,g,m,n,s);
    default:return cooperative<BFloat,ck_tile::GroupedInt8NtTileProblem<16,64,128,1,2,false,false,true,true,1,true>>(args,scales,g,m,n,s);
 }
}
template<bool BF> int dispatch(unsigned route,unsigned legacy,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
 using namespace ck_tile::int8_nt_instances;
 switch(route){
 case 0:return raw<BF>(legacy,args,scales,g,m,n,stream);
 case 1:return wasp<BF,RawM256N256>(args,scales,g,m,n,stream);
 case 2:return wasp<BF,RawM256N256Record1024>(args,scales,g,m,n,stream);
 case 3:return wasp<BF,RawTransposeM256N256>(args,scales,g,m,n,stream);
 case 4:return wasp<BF,RawTransposeM256N256Record1024>(args,scales,g,m,n,stream);
 case 5:return wasp<BF,PackedM256N256>(args,scales,g,m,n,stream);
 case 6:return wasp<BF,PackedM128N256>(args,scales,g,m,n,stream);
 case 7:return wasp<BF,PackedM128N128>(args,scales,g,m,n,stream);
 case 8:return wasp<BF,PackedM64N256>(args,scales,g,m,n,stream);
 case 9:return wasp<BF,PackedM64N64>(args,scales,g,m,n,stream);
 case 10:return wasp<BF,PackedM32N256>(args,scales,g,m,n,stream);
 case 11:return wasp<BF,PackedM32N64>(args,scales,g,m,n,stream);
 case 12:return wasp<BF,PackedM256N384>(args,scales,g,m,n,stream);
 default:return -5;
 }
}
}
extern "C" int ck_tile_int8_scaled_launch_private(unsigned output,unsigned route,unsigned legacy,const void* args,const void* scales,int g,int m,int n,hipStream_t stream){
#ifdef CK_INT8_CACHED_TUNING
 if(route==0 && legacy>=9001 && legacy<=9057)return ck_tile_int8_operand_launch_private(output,legacy,args,scales,g,m,n,stream);
 if(route>=32 && route<=43)return ck_tile_int8_packed_small_launch_private(output,route,args,scales,g,m,n,stream);
 if((route>=13 && route<=21) || (route>=48 && route<=60))return ck_tile_int8_experimental_wasp_launch_private(output,route,args,scales,g,m,n,stream);
#endif
 return output==1?dispatch<true>(route,legacy,args,scales,g,m,n,stream):dispatch<false>(route,legacy,args,scales,g,m,n,stream);
}
