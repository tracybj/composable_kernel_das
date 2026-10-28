// SPDX-License-Identifier: MIT
// Experimental CTA-local K reduction. Acceptance still uses the public selector.
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../small_m_v19/instances.cpp"
#undef ck_tile_grouped_int8_nt_launch

namespace ck_tile {
template <int Split>
struct Int8NtLocalKKernel
{
    using Mmac = WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using Acc = typename Mmac::CVecType;
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto a = args[blockIdx.z].group_karg;
        const index_t tid=get_thread_id(), lane=tid%64, part=tid/64;
        const index_t row=blockIdx.y*16+lane%16, nb=blockIdx.x*16;
        const index_t col=nb+lane%16, ko=(lane/16)*8;
        if(a.M<=0 || a.N<=0 || a.K<=0) return;
        Acc c{};
        const auto load_a = [&](index_t kb) {
            typename Mmac::AVecType v{};
            if(row<a.M && kb+ko+8<=a.K)
                v=*reinterpret_cast<const typename Mmac::AVecType*>(
                    static_cast<const int8_t*>(a.as_ptr[0])+static_cast<long long>(row)*a.stride_As[0]+kb+ko);
            return v;
        };
        const auto load_b = [&](index_t kb) {
            typename Mmac::BVecType v{};
            if(col<a.N && kb+ko+8<=a.K)
                v=*reinterpret_cast<const typename Mmac::BVecType*>(
                    static_cast<const int8_t*>(a.bs_ptr[0])+static_cast<long long>(col)*a.stride_Bs[0]+kb+ko);
            return v;
        };
        index_t kb=part*32;
        auto av=load_a(kb);auto bv=load_b(kb);
        for(;kb+Split*32<a.K;kb+=Split*32)
        {
            auto next_a=load_a(kb+Split*32);
            auto next_b=load_b(kb+Split*32);
            __builtin_amdgcn_sched_barrier(0);
            Mmac{}(c,av,bv);
            av=next_a;bv=next_b;
        }
        if(kb<a.K) Mmac{}(c,av,bv);
        __shared__ int32_t partial[Split][64][4];
        static_for<0,4,1>{}([&](auto i){partial[part][lane][decltype(i)::value]=c[decltype(i)::value];});
        block_sync_lds();
        if(part==0 && row<a.M)
        {
            static_for<0,4,1>{}([&](auto i){
                int32_t total=0;
                static_for<0,Split,1>{}([&](auto s){total+=partial[s][lane][i];});
                const index_t outcol=nb+lane/16+i*4;
                if(outcol<a.N)
                    static_cast<int32_t*>(a.e_ptr)[static_cast<long long>(row)*a.stride_E+outcol]=total;
            });
        }
    }
};
}
namespace {
enum class LocalKInstance : unsigned { split2=6202,split4=6204,split8=6208 };
template<int Split>
int launch_local_k(const void* args,int g,int m,int n,hipStream_t stream)
{
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<Split*64,1>(ck_tile::Int8NtLocalKKernel<Split>{},
            dim3((n+15)/16,(m+15)/16,g),dim3(Split*64),0,
            static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    auto s=reinterpret_cast<hipStream_t>(stream);
    if(!args || g<=0 || g>65535 || m<=0 || n<=0) return -1;
    switch(static_cast<LocalKInstance>(instance_id))
    {
        case LocalKInstance::split2:return launch_local_k<2>(args,g,m,n,s);
        case LocalKInstance::split4:return launch_local_k<4>(args,g,m,n,s);
        case LocalKInstance::split8:return launch_local_k<8>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
