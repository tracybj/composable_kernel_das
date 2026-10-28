// SPDX-License-Identifier: MIT
// Development candidates; no case-ID selection in production.
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "wide_kernel.hpp"
namespace ck_tile {
struct SmallM512Problem : GroupedInt8NtTileProblem<16,32,64,1,2,false,false,true,true,1,true>
{
    static constexpr index_t BK=512;
};
}
namespace {
enum class LayoutInstance:unsigned {
    small_n32_k128_w2=6501,small_n32_k256_w2=6502,small_n32_k256_w1=6503,
    small_n64_k256_w4=6504,small_n32_k512_w2=6505,
    wide_m256_n256_mfirst=6601,wide_m256_n256_nfirst=6602,
    wide_m256_n128_nfirst=6603,wide_m128_n256_nfirst=6604
};
template<class P>
int launch_wide(const void* args,int groups,int m,int n,hipStream_t stream)
{
    dim3 grid(P::MFirst?(m+P::BM-1)/P::BM:(n+P::BN-1)/P::BN,
              P::MFirst?(n+P::BN-1)/P::BN:(m+P::BM-1)/P::BM,groups);
    if constexpr(P::MGroup>0)
    {
        const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
        grid=dim3(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,groups);
    }
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<P::Threads,P::MinBlocks>(ck_tile::GroupedInt8NtWideReadKernel<P>{},
        grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
    auto s=reinterpret_cast<hipStream_t>(stream);
    switch(static_cast<LayoutInstance>(instance_id))
    {
        case LayoutInstance::small_n32_k128_w2:
            return launch<ck_tile::GroupedInt8NtTileProblem<16,32,128,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
        case LayoutInstance::small_n32_k256_w2:
            return launch<ck_tile::GroupedInt8NtTileProblem<16,32,256,1,2,false,false,true,true,1,true>>(args,g,m,n,s);
        case LayoutInstance::small_n32_k256_w1:
            return launch<ck_tile::GroupedInt8NtTileProblem<16,32,256,1,1,false,false,true,true,1,true>>(args,g,m,n,s);
        case LayoutInstance::small_n64_k256_w4:
            return launch<ck_tile::GroupedInt8NtTileProblem<16,64,256,1,4,false,false,true,true,1,true>>(args,g,m,n,s);
        case LayoutInstance::small_n32_k512_w2:
            return launch<ck_tile::SmallM512Problem>(args,g,m,n,s);
        case LayoutInstance::wide_m256_n256_mfirst:
            return launch_wide<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true>>(args,g,m,n,s);
        case LayoutInstance::wide_m256_n256_nfirst:
            return launch_wide<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true,1>>(args,g,m,n,s);
        case LayoutInstance::wide_m256_n128_nfirst:
            return launch_wide<ck_tile::GroupedInt8NtTileProblem<256,128,64,4,2,true,true,true,false,2,true,1>>(args,g,m,n,s);
        case LayoutInstance::wide_m128_n256_nfirst:
            return launch_wide<ck_tile::GroupedInt8NtTileProblem<128,256,64,2,4,true,true,true,false,2,true,1>>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
