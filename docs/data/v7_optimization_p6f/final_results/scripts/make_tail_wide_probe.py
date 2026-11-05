"""CPU prototype justified by the frozen high-G BLAS wide-tile family.

One128x256 eight-wave panel reuses A across twice the N width. All64 lanes
load four BF16 (8B) each for A, filling128x16 exactly; B uses16B/lane for16x256.
This matches the frozen wide Native family's logical-A vector width. No vendor
implementation is copied. GPU comparison remains conditional on tail128 data.
"""
from pathlib import Path
import json
R=Path('/workspace/composable_kernel');I=R/'include/ck_tile/ops/gemm/pipeline';G=R/'example_hcu/ck_tile/19_grouped_gemm';P=Path('/workspace/v7_coverage/p6f')
s=(I/'gemm_pipeline_v7_tail128.hpp').read_text()
for a,b in [('GemmPipelineV7Tail128','GemmPipelineV7Tail256'),('V7Tail128Geometry','V7Tail256Geometry'),('namespace v7tail {','namespace v7tailwide {'),('v7tail::','v7tailwide::'),('} // namespace v7tail','} // namespace v7tailwide'),('kBlockM=128, kBlockN=128','kBlockM=128, kBlockN=256'),('kWaves=4, kThreads=256','kWaves=8, kThreads=512'),('kWaveGroup=2','kWaveGroup=4'),('kLdsBytesA=4096, kLdsBytesB=4096, kLdsBytesPerStage=8192','kLdsBytesA=4096, kLdsBytesB=8192, kLdsBytesPerStage=12288'),('static_assert(TileN == 128)','static_assert(TileN == 256)'),('const int wave_m   = wid >> 1;','const int wave_m   = wid >> 2;'),('const int wave_n   = wid & 1;','const int wave_n   = wid & 3;'),('(uint32_t)(wid * 1024)','(uint32_t)(wid * 512)'),('wid * 32 + (lane >> 1)','wid * 16 + (lane >> 1)'),('const int b_k = (int)tid >> 4;','const int b_k = (int)tid >> 5;'),('const uint32_t octet = (uint32_t)(lane & 15) ^','const uint32_t octet = (uint32_t)(lane & 31) ^'),('rk * 256u','rk * 512u')]:
    assert a in s,a;s=s.replace(a,b)
old='''        const int a_row_global = m_block_start + wid * 16 + (lane >> 1);
        const uint32_t a_kh =
            ((((lane >> 1) & 15) >> 2) & 1) ^ (uint32_t)(lane & 1);'''
new='''        const int a_row_global = m_block_start + wid * 16 + (lane >> 2);
        // Four8B quarters per row; preserve the old row-dependent16B half swap.
        const uint32_t a_kh = (uint32_t)(lane & 3) ^ (2u * ((uint32_t)(lane >> 4) & 1u));'''
assert s.count(old)==1;s=s.replace(old,new).replace('a_kh * 16','a_kh * 8')
import re
s,count=re.subn(r'(?:v7u::)?buffer_load_dwordx4_idxen_lds\((\s*[Ss]\.a_rsrc)',r'v7tailwide::load_a(\1',s)
assert count==3,count
helper='''namespace v7tailwide {
CK_TILE_DEVICE void load_a(v7u::buffer_rsrc_t rsrc, uint64_t pair, uint32_t soffset, uint32_t m0)
{
    asm volatile("s_mov_b32 m0, %[m0]\\n\\t"
                 "buffer_load_dwordx2 %[pair], %[rsrc], %[so] idxen offen lds\\n"
                 :
                 : [pair] "v"(pair), [rsrc] "s"(rsrc), [so] "s"(soffset), [m0] "s"(m0)
                 : "memory");
}
'''
assert s.count('namespace v7tailwide {')==1;s=s.replace('namespace v7tailwide {',helper)
(I/'gemm_pipeline_v7_tail256.hpp').write_text(s)
s=(G/'grouped_gemm_v7_tail_epilogue.hpp').read_text().replace('GroupedGemmV7TailStoreEpilogue','GroupedGemmV7Tail256StoreEpilogue').replace('(wid >> 1) * (int)G::kWaveTileM','(wid >> 2) * (int)G::kWaveTileM').replace('(wid & 1) * (int)G::kWaveTileN','(wid & 3) * (int)G::kWaveTileN')
(G/'grouped_gemm_v7_tail256_epilogue.hpp').write_text(s)
s=(G/'instances/grouped_gemm_v7_tail_launch.hpp').read_text().replace('v7_tail_launch','v7_tail256_launch').replace('grouped_gemm_v7_tail_epilogue.hpp','grouped_gemm_v7_tail256_epilogue.hpp').replace('gemm_pipeline_v7_tail128.hpp','gemm_pipeline_v7_tail256.hpp').replace('GemmConfigV7Tail128','GemmConfigV7Tail256').replace('GemmPipelineV7Tail128<Problem,128>','GemmPipelineV7Tail256<Problem,256>').replace('GroupedGemmV7TailStoreEpilogue','GroupedGemmV7Tail256StoreEpilogue').replace('    static constexpr ck_tile::index_t M_Tile = 128;','    static constexpr ck_tile::index_t M_Tile = 128, N_Tile = 256;\n    static constexpr ck_tile::index_t N_Warp = 4;\n    static constexpr int kBlockPerCu = 1;').replace('ck_tile::make_kernel<256,','ck_tile::make_kernel<512,')
(G/'instances/grouped_gemm_v7_tail256_launch.hpp').write_text(s)
cpp=P/'probe_tailwide.cpp';cpp.write_text(f'''#include "{G/'instances/grouped_gemm_v7_tail256_launch.hpp'}"
extern "C" int probe_tailwide(const void* args, int groups, std::uint64_t tiles, ck_tile::index_t tpg, hipStream_t stream)
{{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return v7_tail256_launch::launch<GemmConfigV7Tail256,Row,Row,Row,ck_tile::bf16_t,ck_tile::bf16_t,float,ck_tile::bf16_t>(args,groups,tiles,tpg,stream);
}}
''')
commands=json.loads((P/'probe_commands.json').read_text());base=commands[0]['command'];base=base[:base.index('-c')]
commands.append(dict(name='tailwide',command=base+['-c',str(cpp),'-o',str(P/'probe_tailwide.o')]))
(P/'probe_wide_commands.json').write_text(json.dumps(commands[-1],indent=2))
# Dedicated probe script input, original three-command manifest preserved.
print('TAIL_WIDE_CPU_PROTOTYPE_PREPARED; GPU comparison conditional on first tail data')
