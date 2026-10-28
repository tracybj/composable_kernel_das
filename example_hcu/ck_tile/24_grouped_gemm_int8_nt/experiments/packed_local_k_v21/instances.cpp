// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../small_m_v19/instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "wave_pipeline.hpp"
namespace ck_tile {
template<int BN,int Split>
struct PackedLocalKKernel
{
    using P=GroupedInt8NtTileProblem<16,BN,64,1,1,false,false,true,true,1,true>;
    using Policy=GroupedInt8NtTilePolicy<P>;
    static constexpr int InputBytes=Split*Policy::OperandBytes;
    static constexpr int OutputBytes=Split*16*BN*4;
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto& original=args[blockIdx.z].group_karg;
        if(original.M<=0 || original.N<=0 || original.K<=0 || original.K%(Split*64)!=0) return;
        const index_t tid=get_thread_id(),part=tid/64,lane=tid%64;
        const index_t mb=blockIdx.y*16,nb=blockIdx.x*BN;
        const UniversalGemmKernelArgs<> a{
            {static_cast<const int8_t*>(original.as_ptr[0])+part*(original.K/Split)},
            {static_cast<const int8_t*>(original.bs_ptr[0])+part*(original.K/Split)},
            {},original.e_ptr,original.M,original.N,original.K/Split,
            {original.stride_As[0]},{original.stride_Bs[0]},{},original.stride_E,1};
        __shared__ __align__(16) int8_t smem[InputBytes>OutputBytes?InputBytes:OutputBytes];
        typename P::Mmac::CVecType c[P::MR][P::NR]{};
        GroupedInt8NtWavePipeline<P>{}(a,mb,nb,smem+part*Policy::OperandBytes,c);
        block_sync_lds();
        auto* partial=reinterpret_cast<int32_t*>(smem);
        static_for<0,P::NR,1>{}([&](auto j){
            static_for<0,4,1>{}([&](auto v){
                constexpr int ni=decltype(j)::value,vi=decltype(v)::value;
                const index_t col=ni*16+lane/16+vi*4;
                partial[(part*16+lane%16)*BN+col]=c[0][ni][vi];
            });
        });
        block_sync_lds();
        for(index_t vec=tid;vec<16*BN/4;vec+=Split*64)
        {
            const index_t row=vec/(BN/4),col=(vec%(BN/4))*4;
            if(mb+row<a.M && nb+col+4<=a.N)
            {
                typename P::CopyVector sum{};
                static_for<0,Split,1>{}([&](auto s){
                    sum+=*reinterpret_cast<const typename P::CopyVector*>(partial+(s*16+row)*BN+col);
                });
                *reinterpret_cast<typename P::CopyVector*>(static_cast<int32_t*>(a.e_ptr)+
                    static_cast<long long>(mb+row)*a.stride_E+nb+col)=sum;
            }
        }
    }
};
}
namespace {
enum class PackedLocalKInstance:unsigned {n128_split2=6302,n96_split4=6304,n64_split4=6306};
template<int BN,int Split>
int launch_packed_local_k(const void* args,int g,int m,int n,hipStream_t stream)
{
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<Split*64,1>(ck_tile::PackedLocalKKernel<BN,Split>{},
            dim3((n+BN-1)/BN,(m+15)/16,g),dim3(Split*64),0,
            static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
    auto s=reinterpret_cast<hipStream_t>(stream);
    switch(static_cast<PackedLocalKInstance>(instance_id))
    {
        case PackedLocalKInstance::n128_split2:return launch_packed_local_k<128,2>(args,g,m,n,s);
        case PackedLocalKInstance::n96_split4:return launch_packed_local_k<96,4>(args,g,m,n,s);
        case PackedLocalKInstance::n64_split4:return launch_packed_local_k<64,4>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
