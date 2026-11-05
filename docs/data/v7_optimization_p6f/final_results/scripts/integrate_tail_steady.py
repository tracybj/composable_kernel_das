"""Run only after cache build finishes; wire independent tail/steady providers."""
from pathlib import Path
R=Path('/workspace/composable_kernel');G=R/'example_hcu/ck_tile/19_grouped_gemm';I=R/'include/ck_tile/ops/gemm'
entries=[('tail128',5013,'tail'),('small_m_k32_steady',5014,'k32'),('small_m_k16_steady',5015,'k16'),('tail256',5016,'tailwide'),('small_m_noslc',5017,'cache_small')]
def edit(p,a,b):
    s=p.read_text();assert s.count(a)==1,(p,a[:80],s.count(a));p.write_text(s.replace(a,b))
for name,num,kind in entries:
    header='grouped_gemm_v7_tail_launch.hpp' if kind=='tail' else 'grouped_gemm_v7_tail256_launch.hpp' if kind=='tailwide' else 'grouped_gemm_v7_small_cache_launch.hpp' if kind=='cache_small' else 'grouped_gemm_v7_steady_launch.hpp'
    ns='v7_tail_launch' if kind=='tail' else 'v7_tail256_launch' if kind=='tailwide' else 'v7_small_cache_launch' if kind=='cache_small' else 'v7_steady_launch'
    config='GemmConfigV7Tail128' if kind=='tail' else 'GemmConfigV7Tail256' if kind=='tailwide' else 'GemmConfigV7SmallCache' if kind=='cache_small' else 'GemmConfigV7Steady<1>' if kind=='k32' else 'GemmConfigV7Steady<0>'
    body=f'''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// P6f independent experimental instance {num}; attribute contract in selector.
#include "{header}"
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_{name}_device_args(
    void* args, int groups, std::uint32_t num_cu, hipStream_t stream,
    std::uint64_t tiles, ck_tile::index_t tiles_per_group)
{{
    (void)num_cu;
    if(args == nullptr || groups <= 0 || groups > ck_tile::kGroupedGemmV7NnMaxGroups || tiles == 0)
        return -1;
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    return {ns}::launch<{config},Row,Row,Row,ck_tile::bf16_t,ck_tile::bf16_t,float,ck_tile::bf16_t>(args,groups,tiles,tiles_per_group,stream);
}}
'''
    (G/f'instances/grouped_gemm_v7_bf16_nn_{name}_device_args.cpp').write_text(body)
    p=G/'grouped_gemm_v7_v8_device_args_provider_sources.txt';p.write_text(p.read_text()+f'instances/grouped_gemm_v7_bf16_nn_{name}_device_args.cpp\n')
p=I/'grouped_gemm_selector.hpp'
edit(p,'    std::int32_t bf16_v7_nn_cache_walk = 0;','    std::int32_t bf16_v7_nn_cache_walk = 0;\n    bool enable_bf16_v7_nn_tail128 = false;\n    std::int32_t bf16_v7_nn_tail_tile_n = 128;')
edit(p,'    v7_bf16_nn_cache_noslc_m01_5 = 5012,','    v7_bf16_nn_cache_noslc_m01_5 = 5012,\n'+''.join(f'    v7_bf16_nn_{name} = {num},\n' for name,num,_ in entries).rstrip())
edit(p,'std::array<GroupedGemmCandidateIdentity, 31>','std::array<GroupedGemmCandidateIdentity, 36>')
s=p.read_text();a=s.index('        {GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5,');b=s.index('},',a)+2;needle=s[a:b]
edit(p,needle,needle+''.join(f'\n        {{GroupedGemmInstanceId::v7_bf16_nn_{name},\n         "v7.nn_b.{name}", GroupedGemmCandidateFamily::v7}},' for name,_,_ in entries))
edit(p,'    v7_bf16_nn_cache_noslc_m01_5,\n};','    v7_bf16_nn_cache_noslc_m01_5,\n'+''.join(f'    v7_bf16_nn_{name},\n' for name,_,_ in entries)+'};')
needle='    // Experimental small-M NN route. Whole N tiles only, so N=1232 stays'
insert='''    // Independent vector-aligned N tail. Neither the original no-column-
    // predicate epilogue nor its contract changes. B loads and C stores both
    // mask complete eight-BF16 vectors; reject smaller alignment/ragged inputs.
    const auto tail_tile_n = policy.bf16_v7_nn_tail_tile_n;
    if(policy.enable_bf16_v7_nn && policy.enable_bf16_v7_nn_tail128 &&
       (tail_tile_n == 128 || tail_tile_n == 256) &&
       problem.arch == GroupedGemmArch::gfx936 && is_bf16 && is_nn &&
       problem.m.kind == GroupedGemmDimensionKind::common && !has_variable_k &&
       !problem.group_lengths_are_ragged && !problem.may_have_empty_groups &&
       problem.group_count <= kGroupedGemmV7NnMaxGroups && m <= 128 &&
       k >= 64 && k <= 4096 && k % 16 == 0 && n >= 128 && n % 8 == 0 && n % 128 != 0 &&
       m <= std::int64_t{2147483647} / (2 * k) &&
       n <= std::int64_t{2147483647} / (2 * k) &&
       grouped_gemm_v7_experiment_tile_bound(problem, 128, tail_tile_n) <= kGroupedGemmV7MaxTiles)
        return tail_tile_n == 256
            ? GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_tail256,
                                   GroupedGemmSelectionReason::v7_bf16_nn_tail256}
            : GroupedGemmSelection{GroupedGemmInstanceId::v7_bf16_nn_tail128,
                                   GroupedGemmSelectionReason::v7_bf16_nn_tail128};
'''
edit(p,needle,insert+needle)
edit(p,'const bool k32_experiment = experiment == 2 || experiment == 4;','const bool k32_experiment = experiment == 2 || experiment == 4 || experiment == 5;')
edit(p,'policy.enable_bf16_v7_nn && experiment >= 1 && experiment <= 4 &&','policy.enable_bf16_v7_nn && experiment >= 1 && experiment <= 7 &&\n       (experiment < 5 || m <= 64) &&')
needle='''        case 4: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32n64,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k32n64};'''
edit(p,needle,needle+'''
        case 5: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k32_steady,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k32_steady};
        case 6: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_k16_steady,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_k16_steady};
        case 7: return {GroupedGemmInstanceId::v7_bf16_nn_small_m_noslc,
                         GroupedGemmSelectionReason::v7_bf16_nn_small_m_noslc};''')
p=I/'grouped_gemm_device_args.h'
needle='    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_CACHE_NOSLC_M01_5_V1 = 5012,'
edit(p,needle,needle+'\n'+''.join(f'    CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_{name.upper()}_V1 = {num},\n' for name,num,_ in entries).rstrip())
p=G/'grouped_gemm_device_args.cpp'
edit(p,"if(cache_walk != nullptr && cache_walk[1] == '\\0' &&", "if(cache_walk != nullptr && cache_walk[0] != '\\0' && cache_walk[1] == '\\0' &&")
needle='extern "C" int ck_tile_hcu_grouped_gemm_v8_bf16_tn_device_args('
edit(p,needle,''.join(f'extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_{name}_device_args(\n    void*, int, std::uint32_t, hipStream_t, std::uint64_t, std::int32_t);\n' for name,_,_ in entries)+needle)
needle='    // Local experimental adapter: no public ABI flags or shared provider header.'
edit(p,needle,'''    const char* tail_enable = std::getenv("CK_TILE_GROUPED_GEMM_ENABLE_BF16_V7_NN_N_TAIL");
    const char* tail_disable = std::getenv("CK_TILE_GROUPED_GEMM_DISABLE_BF16_V7_NN_N_TAIL");
    policy.enable_bf16_v7_nn_tail128 = tail_enable != nullptr && (tail_enable[0] == '1' || tail_enable[0] == '2') && tail_enable[1] == '\\0' &&
        !(tail_disable != nullptr && tail_disable[0] == '1' && tail_disable[1] == '\\0');
    policy.bf16_v7_nn_tail_tile_n = tail_enable != nullptr && tail_enable[0] == '2' ? 256 : 128;
'''+needle)
edit(p,"experiment[0] <= '4'", "experiment[0] <= '7'")
edit(p,'    policy.bf16_v7_nn_cache_walk = 0;','    policy.bf16_v7_nn_cache_walk = 0;\n    policy.enable_bf16_v7_nn_tail128 = false;')
needle='    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_small_m_drain:\n#if'
insert=''
for name,_,kind in entries:
    tm=128 if kind.startswith('tail') else 64
    tn=256 if kind=='tailwide' else 128
    lookup='static_cast<ck_tile::index_t>(tiles / groups)' if kind.startswith('tail') else '(groups > ck_tile::kGroupedGemmV7ScalarScanMaxGroups ? static_cast<ck_tile::index_t>(tiles / groups) : 0)'
    insert+=f'''    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:
#if defined(CK_TILE_GROUPED_GEMM_DEVICE_ARGS_HAS_V7_V8_BF16)
        if(dtype == ck_tile::GroupedGemmDataType::bf16 && layout == ck_tile::GroupedGemmLayout::nn)
        {{
            const auto tiles = static_cast<std::uint64_t>(ck_tile::grouped_gemm_v7_experiment_tile_bound(problem, {tm}, {tn}));
            const std::uint64_t slots = launch.num_cu > 0 ? launch.num_cu : 1;
            const auto tile_slots = (tiles + slots - 1) / slots * slots;
            const auto tpg = {lookup};
            return ck_tile_hcu_grouped_gemm_v7_bf16_nn_{name}_device_args(device_args, groups, launch.num_cu, stream, tile_slots, tpg);
        }}
#endif
        break;
'''
edit(p,needle,insert+needle)
needle='    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5:\n        architecture_mask'
edit(p,needle,'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5:\n'+''.join(f'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:\n' for name,_,_ in entries)+'        architecture_mask')
p=G/'instances/grouped_gemm_device_args_provider.hpp'
needle='    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_cache_noslc_m01_5:'
edit(p,needle,needle+'\n'+''.join(f'    case ck_tile::GroupedGemmInstanceId::v7_bf16_nn_{name}:\n' for name,_,_ in entries).rstrip())
p=G/'grouped_gemm_device_args_query_test.cpp'
edit(p,'std::array<CandidateExpected, 31>','std::array<CandidateExpected, 36>')
s=p.read_text();a=s.index('        {CK_TILE_HCU_GROUPED_GEMM_INSTANCE_V7_BF16_NN_CACHE_NOSLC_M01_5_V1,');b=s.index('},',a)+2;needle=s[a:b]
edit(p,needle,needle+''.join('\n'+needle.replace('CACHE_NOSLC_M01_5',name.upper()) for name,_,_ in entries))
p=G/'grouped_gemm_selector_test.cpp'
edit(p,'for(int invalid : {-1,0,5,128})','for(int invalid : {-1,0,8,128})')
p=G/'CMakeLists.txt'
edit(p,'    # Separate manifest: includes opt-in cache/walk and small-M experiments (5005..5012).','    # Separate manifest: includes opt-in cache/walk, independent N-tail, steady/drain and limited small-M cache control (5005..5017).')
print('TAIL_STEADY_INTEGRATED new_ids=5013,5014,5015,5016,5017 default_off=true')
