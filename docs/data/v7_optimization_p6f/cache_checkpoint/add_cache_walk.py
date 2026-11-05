"""Integrate two new CK-owned M01=5 providers; leave old pipelines untouched."""
from pathlib import Path
R=Path('/workspace/composable_kernel'); G=R/'example_hcu/ck_tile/19_grouped_gemm'; I=R/'include/ck_tile/ops/gemm'
def edit(p,a,b):
    s=p.read_text();assert s.count(a)==1,(p,a[:120],s.count(a));p.write_text(s.replace(a,b))
entries=[('cache_m01_5',5011,False),('cache_noslc_m01_5',5012,True)]
for name,num,noslc in entries:
    source=G/('instances/grouped_gemm_v7_bf16_nn_a_noslc_device_args.cpp' if noslc else 'instances/grouped_gemm_v7_bf16_nn_device_args.cpp')
    body=source.read_text()
    old='a_noslc' if noslc else ''
    body=body.replace('ck_tile_hcu_grouped_gemm_v7_bf16_nn'+('_'+old if old else '')+'_device_args','ck_tile_hcu_grouped_gemm_v7_bf16_nn_'+name+'_device_args')
    body=body.replace('launch<GemmConfigV7NnBf16,','launch<GemmConfigV7Uniform<ck_tile::bf16_t, CK_TILE_PIPELINE_COMPUTE_V7, 5>,')
    (G/f'instances/grouped_gemm_v7_bf16_nn_{name}_device_args.cpp').write_text('// P6f experimental M01=5, instance '+str(num)+'.\n'+body)
    p=G/'grouped_gemm_v7_v8_device_args_provider_sources.txt'
    p.write_text(p.read_text()+f'instances/grouped_gemm_v7_bf16_nn_{name}_device_args.cpp\n')
p=I/'grouped_gemm_selector.hpp'
edit(p,'    bool enable_bf16_v7_nn_a_noslc = false; // Independent cache-policy experiment.','    bool enable_bf16_v7_nn_a_noslc = false; // Independent cache-policy experiment.\n    // P6f diagnostic factor: 0 keeps the old policy; 1/5/8 set a tile walk.\n    std::int32_t bf16_v7_nn_cache_walk = 0;')
edit(p,'    v7_bf16_nn_small_m_k32n64 = 5010,','    v7_bf16_nn_small_m_k32n64 = 5010,\n'+''.join(f'    v7_bf16_nn_{name} = {num},\n' for name,num,_ in entries).rstrip())
edit(p,'std::array<GroupedGemmCandidateIdentity, 29>','std::array<GroupedGemmCandidateIdentity, 31>')
needle='        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64,\n         "v7.nn_b.small_m.64x64x32.s4", GroupedGemmCandidateFamily::v7},'
# Obtain the actual existing canonical suffix instead of assuming it.
s=p.read_text();a=s.index('        {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64,');b=s.index('},',a)+2;needle=s[a:b]
edit(p,needle,needle+''.join(f'\n        {{GroupedGemmInstanceId::v7_bf16_nn_{name},\n         "v7.nn_b.uniform_d2l.256x256x16.{name}", GroupedGemmCandidateFamily::v7}},' for name,_,_ in entries))
edit(p,'    v7_bf16_nn_small_m_k32n64,\n};','    v7_bf16_nn_small_m_k32n64,\n'+''.join(f'    v7_bf16_nn_{name},\n' for name,_,_ in entries)+'};')
needle='        // Experimental cache-only variant: keep the original tile walk.'
insert='''        // P6f factor isolation. An explicit opt-in controls only a valid common
        // NN V7 contract; this is not an ID/shape dispatch override.
        if((policy.bf16_v7_nn_cache_walk == 1 || policy.bf16_v7_nn_cache_walk == 5 ||
            policy.bf16_v7_nn_cache_walk == 8) &&
           problem.m.kind == GroupedGemmDimensionKind::common &&
           !problem.group_lengths_are_ragged && k <= 1073741823 &&
           m <= std::int64_t{2147483647} / (2 * k) &&
           n <= std::int64_t{2147483647} / (2 * k))
        {
            const bool no_slc = policy.enable_bf16_v7_nn_a_noslc;
            if(policy.bf16_v7_nn_cache_walk == 5)
                return no_slc
                    ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,
                                           GroupedGemmSelectionReason::v7_bf16_nn_cache_noslc_m01_5}
                    : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_cache_m01_5,
                                           GroupedGemmSelectionReason::v7_bf16_nn_cache_m01_5};
            if(policy.bf16_v7_nn_cache_walk == 8)
                return no_slc
                    ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_a_noslc_m01_8,
                                           GroupedGemmSelectionReason::v7_bf16_nn_a_noslc_m01_8}
                    : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_m01_8,
                                           GroupedGemmSelectionReason::v7_bf16_nn_m01_8};
            return no_slc
                ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_a_noslc,
                                       GroupedGemmSelectionReason::v7_bf16_nn_a_noslc}
                : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn,
                                       GroupedGemmSelectionReason::v7_bf16_nn};
        }
'''
edit(p,needle,insert+needle)
p=I/'grouped_gemm_device_args.h'
edit(p,'    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_SMALL_M_K32N64_V1 = 5010,','    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_SMALL_M_K32N64_V1 = 5010,\n'+''.join(f'    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_{name.upper()}_V1 = {num},\n' for name,num,_ in entries).rstrip())
p=G/'grouped_gemm_device_args.cpp'
needle='extern "C" int ck_tile_hcu_grouped_gemm_v8_bf16_tn_device_args('
edit(p,needle,''.join(f'extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_{name}_device_args(\n    void*, int, std::uint32_t, hipStream_t, std::uint64_t, std::int32_t);\n' for name,_,_ in entries)+needle)
needle='    // Local experimental adapter: no public ABI flags or shared provider header.'
edit(p,needle,'''    const char* cache_walk = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_CACHE_WALK");
    const char* cache_walk_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_CACHE_WALK");
    if(cache_walk != nullptr && cache_walk[1] == '\\0' &&
       (cache_walk[0] == '1' || cache_walk[0] == '5' || cache_walk[0] == '8') &&
       !(cache_walk_disable != nullptr && cache_walk_disable[0] == '1' && cache_walk_disable[1] == '\\0'))
        policy.bf16_v7_nn_cache_walk = cache_walk[0] - '0';
'''+needle)
edit(p,'    policy.bf16_v7_nn_small_m_experiment = 0;','    policy.bf16_v7_nn_small_m_experiment = 0;\n    policy.bf16_v7_nn_cache_walk = 0;')
needle='    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_drain:\n#if'
insert=''
for name,_,_ in entries:
    insert+=f'''    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 && layout == ck_tile::GroupedGemmLayout::nn)
            return ck_tile_hcu_grouped_gemm_v7_bf16_nn_{name}_device_args(
                device_args, groups, launch.num_cu, stream,
                v7_uniform_launch::block_id_range(problem, launch.num_cu),
                v7_uniform_launch::tiles_per_group(problem));
#endif
        break;
'''
edit(p,needle,insert+needle)
needle='    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64:\n        architecture_mask'
edit(p,needle,'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64:\n'+''.join(f'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:\n' for name,_,_ in entries)+'        architecture_mask')
p=G/'instances/grouped_gemm_device_args_provider.hpp'
edit(p,'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64:','    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64:\n'+''.join(f'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:\n' for name,_,_ in entries).rstrip())
p=G/'grouped_gemm_device_args_query_test.cpp'
edit(p,'std::array<CandidateExpected, 29>','std::array<CandidateExpected, 31>')
s=p.read_text();a=s.index('        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_SMALL_M_K32N64_V1,');b=s.index('},',a)+2;needle=s[a:b]
edit(p,needle,needle+''.join('\n'+needle.replace('SMALL_M_K32N64',name.upper()) for name,_,_ in entries))
p=G/'CMakeLists.txt'
edit(p,'    # Separate manifest: includes opt-in 5005/5006 cache and 5007..5010 small-M experiments.','    # Separate manifest: includes opt-in cache/walk and small-M experiments (5005..5012).')
print('CACHE_WALK_INTEGRATED new_ids=5011,5012 default_off=true')
