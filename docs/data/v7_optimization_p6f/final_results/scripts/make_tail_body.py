"""Create independent CK tail body; do not alter accepted V7 epilogues."""
from pathlib import Path
R=Path('/workspace/composable_kernel');I=R/'include/ck_tile/ops/gemm/pipeline';G=R/'example_hcu/ck_tile/19_grouped_gemm'
s=(I/'gemm_pipeline_v7_square128.hpp').read_text()
s=s.replace('GemmPipelineV7Square128','GemmPipelineV7Tail128')
s=s.replace('buffer_load_dwordx4_idxen_lds_t(', 'buffer_load_dwordx4_idxen_lds(')
s=s.replace('V7Square128Geometry','V7Tail128Geometry').replace('namespace v7sq {','namespace v7tail {').replace('v7sq::','v7tail::').replace('} // namespace v7sq','} // namespace v7tail')
old='const uint32_t index = octet < (uint32_t)(G::kBlockN / 8) ? (uint32_t)b_k : 0xffffffffu;'
new='''// Contract N%8==0: a complete BF16 octet is either inside this
        // row or zero-filled by invalid idxen. Both load and store mask N.
        const uint32_t index = col + 8u <= (uint32_t)p.n ? (uint32_t)b_k : 0xffffffffu;'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('// Experimental logical128x128x16, four waves, existing32KB ring.','// Independent N-tail128x128x16: B load and C store both bound columns.\n// N%8==0 preserves full-vector memory operations; original V7 is unchanged.')
(I/'gemm_pipeline_v7_tail128.hpp').write_text(s)
s=(G/'grouped_gemm_v7_small_m_epilogue.hpp').read_text().replace('GroupedGemmV7SmallMStoreEpilogue','GroupedGemmV7TailStoreEpilogue')
crop_from='''        for(index_t mF = 0; mF < G::kMTilesPerWave; ++mF)
        {
'''
crop_to='''        for(index_t mF = 0; mF < G::kMTilesPerWave; ++mF)
        {
            // Slabs belong to individual waves and this loop has no block
            // barrier. Entire invalid16-row fragments need no conversion or
            // LDS shuffle; the initial global/LDS/block drain stays outside.
            if(p.m_start + wave_m_start + mF * 16 >= p.m) break;
'''
assert s.count(crop_from)==1;s=s.replace(crop_from,crop_to)
old='''                if(row < p.m)
                {
                    const index_t col0 =
                        p.n_start + wave_n_start + (index_t)u * (16 / kE);
                    *reinterpret_cast<u32x4*>(&d_out[row * d_stride + col0]) = vj[j];
                }'''
new='''                const index_t col0 =
                    p.n_start + wave_n_start + (index_t)u * (16 / kE);
                // Contract N%8==0 for BF16: never write a partial vector or
                // spill the last tile into the next row/group/guard.
                if(row < p.m && col0 + (16 / kE) <= p.n)
                    *reinterpret_cast<u32x4*>(&d_out[row * d_stride + col0]) = vj[j];'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('// N must be a whole logical tile; only rows are predicated. For N=64 the row','// N must be an eight-element BF16 multiple; rows and columns are predicated.\n// For N=64 the row')
(G/'grouped_gemm_v7_tail_epilogue.hpp').write_text(s)
s=(G/'instances/grouped_gemm_v7_small_experiment_launch.hpp').read_text()
a=s.index('namespace v7_small_experiment_launch {');s=s[a:]
s=s.replace('v7_small_experiment_launch','v7_tail_launch').replace('typename Config::template ExperimentPipeline<Problem>','ck_tile::GemmPipelineV7Tail128<Problem,128>').replace('GroupedGemmV7SmallMStoreEpilogue','GroupedGemmV7TailStoreEpilogue')
prefix='''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once
#include "grouped_gemm_v7_small_m_launch.hpp"
#include "../grouped_gemm_v7_tail_epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_v7_tail128.hpp"
struct GemmConfigV7Tail128 : GemmConfigV7SmallM<128>
{
    static constexpr ck_tile::index_t M_Tile = 128;
};
'''
(G/'instances/grouped_gemm_v7_tail_launch.hpp').write_text(prefix+s)
print('TAIL_BODY_CREATED independent=true load_mask=true store_mask=true')
