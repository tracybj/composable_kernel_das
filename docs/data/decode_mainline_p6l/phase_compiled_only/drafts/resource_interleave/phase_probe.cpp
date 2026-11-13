#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "/workspace/v7_coverage/p6l_decode_mainline/run_20261112T025323Z/drafts/resource_interleave/gemm_pipeline_decode_nn_resource_interleave.hpp"
namespace ck_tile {
template<int Rows, bool Wide, bool AllFirst>
struct DecodeNnPhaseProbeKernel
{
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* args,
                                   int groups, int tiles_per_group) const
    {
        const int block = get_block_1d_id();
        const int group = block / tiles_per_group;
        if(group >= groups) return;
        const auto* descriptors = reinterpret_cast<const GemmTransKernelArgImpl<0>*>(
            cast_pointer_to_generic_address_space(args));
        const auto& a = descriptors[group].group_karg;
        decode_nn_mainline::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E,
            (block - group * tiles_per_group) * 128};
        if constexpr(Wide)
        {
            if constexpr(AllFirst) decode_nn_resource_interleave::wide_detail::run<Rows,32,4,true>(p);
            else decode_nn_resource_interleave::run_wide<Rows>(p);
        }
        else
        {
            if constexpr(AllFirst) decode_nn_resource_interleave::compact_detail::run<Rows,true>(p);
            else decode_nn_resource_interleave::run_compact<Rows>(p);
        }
    }
};

template<int Rows,bool Wide,bool AllFirst>
CK_TILE_HOST int phase_probe_launch(void* args,int groups,hipStream_t stream,uint64_t tiles,int tpg)
{
    if constexpr(Wide)
    {
        using G=decode_nn_cooperative_wide::Geometry<Rows,32,4>;
        const auto kernel=make_kernel<512,1>(DecodeNnPhaseProbeKernel<Rows,Wide,AllFirst>{},dim3(static_cast<unsigned>(tiles)),dim3(512),G::LdsBytes,cast_pointer_to_constant_address_space(args),groups,tpg);
        launch_kernel(stream_config{stream},kernel);
    }
    else
    {
        using G=decode_nn_cooperative_compact::Geometry<Rows>;
        const auto kernel=make_kernel<256,2>(DecodeNnPhaseProbeKernel<Rows,Wide,AllFirst>{},dim3(static_cast<unsigned>(tiles)),dim3(256),G::LdsBytes,cast_pointer_to_constant_address_space(args),groups,tpg);
        launch_kernel(stream_config{stream},kernel);
    }
    return 0;
}
} // namespace ck_tile
extern "C" int p6l_phase_probe_not_executed(void* args,int groups,hipStream_t stream,uint64_t tiles,int tpg,int mode)
{
    switch(mode)
    {
    case 0: return ck_tile::phase_probe_launch<1,false,false>(args,groups,stream,tiles,tpg);
    case 1: return ck_tile::phase_probe_launch<1,false,true>(args,groups,stream,tiles,tpg);
    case 2: return ck_tile::phase_probe_launch<2,false,false>(args,groups,stream,tiles,tpg);
    case 3: return ck_tile::phase_probe_launch<2,false,true>(args,groups,stream,tiles,tpg);
    case 4: return ck_tile::phase_probe_launch<4,false,false>(args,groups,stream,tiles,tpg);
    case 5: return ck_tile::phase_probe_launch<4,false,true>(args,groups,stream,tiles,tpg);
    case 6: return ck_tile::phase_probe_launch<2,true,false>(args,groups,stream,tiles,tpg);
    case 7: return ck_tile::phase_probe_launch<2,true,true>(args,groups,stream,tiles,tpg);
    case 8: return ck_tile::phase_probe_launch<4,true,false>(args,groups,stream,tiles,tpg);
    case 9: return ck_tile::phase_probe_launch<4,true,true>(args,groups,stream,tiles,tpg);
    default:return -1;
    }
}
