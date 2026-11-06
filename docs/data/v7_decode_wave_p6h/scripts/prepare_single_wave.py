from pathlib import Path
import json
repo=Path('/workspace/composable_kernel');root=Path('/workspace/v7_coverage/p6h_decode_arch')
for name in ['build_trial04.json','build_trial04.log']:
 p=root/name
 if p.exists():p.rename(root/name.replace('trial04','trial04_prepare_failed'))
s=(root/'draft_single_wave.hpp').read_text();start=s.index('// Independent decode dataflow');end=s.index('#pragma clang diagnostic push',start)
s=s[:start]+'''// Independent decode dataflow: one wave owns the entire16/32/64x128 tile.
// A stays in VGPR; B uses private4KiB LDS panels. A is reused across all N
// fragments, B across all M fragments, and global B loads cover256B rows.
// There is no inter-wave communication, K reduction or CTA barrier.
'''+s[end:]
new_provider='''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp"

namespace ck_tile {
template<int Rows, int Depth>
struct DecodeNnWaveKernel
{
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* args,
                                   int groups, int tiles_per_group) const
    {
        const int block = get_block_1d_id();
        const int group = block / tiles_per_group;
        if(group >= groups) return;
        const auto* desc = reinterpret_cast<const GemmTransKernelArgImpl<0>*>(
            cast_pointer_to_generic_address_space(args));
        const auto& a = desc[group].group_karg;
        const int tile = block - group * tiles_per_group;
        decode_nn_wave::Params p{
            reinterpret_cast<const uint16_t*>(a.as_ptr[0]),
            reinterpret_cast<const uint16_t*>(a.bs_ptr[0]),
            reinterpret_cast<uint16_t*>(a.e_ptr),
            a.M, a.N, a.K, a.stride_As[0], a.stride_Bs[0], a.stride_E, 0, tile * 128};
        decode_nn_wave::run<Rows, Depth>(p);
    }
};
template<int Rows, int Depth>
CK_TILE_HOST int launch_decode_nn(void* args, int groups, hipStream_t stream,
                                  uint64_t tiles, int tiles_per_group)
{
    const auto kernel = make_kernel<64, 4>(
        DecodeNnWaveKernel<Rows, Depth>{}, dim3(static_cast<unsigned>(tiles)), dim3(64),
        Depth * 4096, cast_pointer_to_constant_address_space(args), groups, tiles_per_group);
    launch_kernel(stream_config{stream}, kernel);
    return 0;
}
template<int Rows>
CK_TILE_HOST int launch_decode_depth(void* args, int groups, hipStream_t stream,
                                     uint64_t tiles, int tiles_per_group, int k)
{
    return k >= 128 && k % 128 == 0
        ? launch_decode_nn<Rows, 8>(args, groups, stream, tiles, tiles_per_group)
        : launch_decode_nn<Rows, 2>(args, groups, stream, tiles, tiles_per_group);
}
} // namespace ck_tile

extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_wave_device_args(
    void* args, int groups, hipStream_t stream, uint64_t tiles, int tiles_per_group, int m, int k)
{
    if(args == nullptr || groups <= 0 || groups > 1024 || tiles == 0 ||
       tiles > (uint64_t{1} << 30) || tiles_per_group <= 0 || m < 1 || m > 64)
        return -1;
    if(m <= 16) return ck_tile::launch_decode_depth<1>(args, groups, stream, tiles, tiles_per_group, k);
    if(m <= 32) return ck_tile::launch_decode_depth<2>(args, groups, stream, tiles, tiles_per_group, k);
    return ck_tile::launch_decode_depth<4>(args, groups, stream, tiles, tiles_per_group, k);
}
'''
p=repo/'example_hcu/ck_tile/19_grouped_gemm/grouped_gemm_device_args.cpp';adapter=p.read_text();old='''extern "C" int ck_tile_hcu_grouped_gemm_bf16_nn_decode_wave_device_args(
    void*, int, hipStream_t, std::uint64_t, int, int);''';assert old in adapter;adapter=adapter.replace(old,old[:-2]+', int);')
old='const auto tpg = static_cast<int>(((problem.m.extent + 15) / 16) * (problem.n.extent / 128));';assert adapter.count(old)==1;adapter=adapter.replace(old,'const auto tpg = static_cast<int>(problem.n.extent / 128);')
old='device_args, groups, stream, static_cast<std::uint64_t>(tpg) * groups, tpg, static_cast<int>(problem.k.extent));';assert old in adapter;adapter=adapter.replace(old,'device_args, groups, stream, static_cast<std::uint64_t>(tpg) * groups, tpg, static_cast<int>(problem.m.extent), static_cast<int>(problem.k.extent));')
for k in range(16):
 for chunk in range(16):
  physical=k*256+((chunk^((k&1)*4))*16);packet=k//4;lane=(k%4)*16+(chunk^((k&1)*4));assert physical==packet*1024+lane*16
for rows in [1,2,4]:
 allout=[]
 for m in range(rows):
  staged={}
  for lane in range(64):
   r=lane&15;q=lane>>4
   for n in range(8):
    for j in range(4):
     col=q+n*16+j*4;b=col*2;addr=r*256+(((b>>4)^(r&7))<<4)+(b&15);assert addr not in staged;staged[addr]=(m*16+r,col)
  for lane in range(64):
   for packet in range(4):
    t=lane+packet*64;r=t>>4;u=t&15;addr=r*256+((u^(r&7))<<4);allout.extend(staged[addr+j*2] for j in range(8))
 assert sorted(allout)==[(m,n) for m in range(rows*16) for n in range(128)]
(repo/'include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_wave.hpp').write_text(s)
(repo/'example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_wave_device_args.cpp').write_text(new_provider);p.write_text(adapter)
(root/'trial04_hypothesis.json').write_text(json.dumps(dict(design='Single wave owns16/32/64x128 output tile, completeK, stage8 forK%128 else2. Same A swizzle and no-global-load drain. No duplicatedA across N waves; no duplicatedB across M16 CTAs.',evidence='Depth8 ID772 HBM9200MB vs baseline2527MB with quarterMMOP and8xCTA; restore fullM reuse and256B global B row coalescing.',scope='Same6+3accuracy and22representative contrasts, no100extension until assessed',canonical_note='Initial unreleased .s2 suffix will be generalized before delivery.'),indent=2))
(root/'single_wave_address_coverage.json').write_text(json.dumps(dict(result='PASS',rows=[16,32,64],B_panel_B=4096,one_wave=True,note='CPU address coverage only; GPU validation remains required'),indent=2));print('SINGLE_WAVE_ADDRESS_COVERAGE_PASS')
