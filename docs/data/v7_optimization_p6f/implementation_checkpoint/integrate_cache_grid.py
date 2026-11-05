"""Run only if the complete predeclared12-case cache domain supports adoption."""
from pathlib import Path
R=Path('/workspace/composable_kernel');I=R/'include/ck_tile/ops/gemm';G=R/'example_hcu/ck_tile/19_grouped_gemm'
def edit(p,a,b):
    s=p.read_text();assert s.count(a)==1,(p,a[:80],s.count(a));p.write_text(s.replace(a,b))
edit(I/'grouped_gemm_selector.hpp','    std::int32_t bf16_v7_nn_cache_walk = 0;',
     '    std::int32_t bf16_v7_nn_cache_walk = 0;\n    bool enable_bf16_v7_nn_cache_grid = false;')
needle='        // Experimental cache-only variant: keep the original tile walk.'
insert='''        // P6f opt-in bounded cache policy, validated as a complete attribute
        // domain. No caller tile knowledge or per-shape lookup is required.
        if(policy.enable_bf16_v7_nn_cache_grid &&
           problem.m.kind == GroupedGemmDimensionKind::common &&
           !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
           m > 1024 && m <= 2048 && n > 4096 && k >= 3072 && k <= 4096 &&
           m <= std::int64_t{2147483647} / (2 * k) &&
           n <= std::int64_t{2147483647} / (2 * k))
            return {GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,
                    GroupedGemmSelectionReason::v7_bf16_nn_cache_noslc_m01_5};
'''
edit(I/'grouped_gemm_selector.hpp',needle,insert+needle)
needle='    // Local experimental adapter: no public ABI flags or shared provider header.'
insert='''    const char* cache_grid = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_GRID");
    const char* cache_grid_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_GRID");
    policy.enable_bf16_v7_nn_cache_grid = cache_grid != nullptr && cache_grid[0] == '1' && cache_grid[1] == '\\0' &&
        !(cache_grid_disable != nullptr && cache_grid_disable[0] == '1' && cache_grid_disable[1] == '\\0');
'''
edit(G/'grouped_gemm_device_args.cpp',needle,insert+needle)
edit(G/'grouped_gemm_device_args.cpp','    policy.bf16_v7_nn_cache_walk = 0;',
     '    policy.bf16_v7_nn_cache_walk = 0;\n    policy.enable_bf16_v7_nn_cache_grid = false;')
print('CACHE_GRID_INTEGRATED default_off=true attribute_domain=8dataset_cases boundary_controls=4')
