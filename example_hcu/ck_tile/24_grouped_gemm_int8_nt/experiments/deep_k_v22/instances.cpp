// SPDX-License-Identifier: MIT
// Small-M experiment: trade a larger LDS tile for fewer K-loop barriers.
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../small_m_v19/instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
namespace ck_tile {
template<int N,int K,int WavesN>
struct SmallMDeepKProblem : GroupedInt8NtTileProblem<16,N,64,1,WavesN,false,false,true,true,1,true>
{
    static constexpr index_t BK=K;
    static_assert(K==128 || K==256);
};
}
namespace {
enum class DeepKInstance:unsigned {n64_k128_w2=6401,n128_k128_w2=6402,
    n64_k256_w2=6403,n128_k256_w4=6404,n128_k256_w2=6405};
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
    auto s=reinterpret_cast<hipStream_t>(stream);
    switch(static_cast<DeepKInstance>(instance_id))
    {
        case DeepKInstance::n64_k128_w2:
            return launch<ck_tile::SmallMDeepKProblem<64,128,2>>(args,g,m,n,s);
        case DeepKInstance::n128_k128_w2:
            return launch<ck_tile::SmallMDeepKProblem<128,128,2>>(args,g,m,n,s);
        case DeepKInstance::n64_k256_w2:
            return launch<ck_tile::SmallMDeepKProblem<64,256,2>>(args,g,m,n,s);
        case DeepKInstance::n128_k256_w4:
            return launch<ck_tile::SmallMDeepKProblem<128,256,4>>(args,g,m,n,s);
        case DeepKInstance::n128_k256_w2:
            return launch<ck_tile::SmallMDeepKProblem<128,256,2>>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
