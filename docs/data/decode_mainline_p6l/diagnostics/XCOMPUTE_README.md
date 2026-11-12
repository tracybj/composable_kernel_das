# 第一批：当前CK与冻结算法Native GroupGEMM对照

本包覆盖预先固定的4条代表shape，每条有当前最佳CK和Native两份含SQTT的perf。物理GPU4串行，原始精度与同轮计时另见diagnostic_analysis.csv；profiler时长不作为正式性能成绩。

## 打开与观察

1. 完整解压，保留captures内目录和文件的原始名字与相对位置。用XCompute File → Open打开指定.perf。
2. 在Result选择下面列出的dispatch、kernel；XCD选0。先Summary核对grid、block和资源，再Source搜索指令，最后Wavefronts观察同一wave的执行顺序。
3. Source先将双滑块恢复全范围；比较主循环指令的Hit Count、等待依赖和寄存器，不把不同wave累计Latency相加当墙钟。Wavefronts先缩到全kernel，再放大中段重复K循环，随后查看开头预热与末尾排空。
4. 重点反馈：ds_read与MMAC之间是否有独立计算、vmcnt/lgkmcnt具体依赖什么、barrier前后四个wave是否长期错开。等待长条或MMAC利用率低本身不构成单一根因。

本机已检查文件SHA、目标dispatch、实际代码反汇编和有效SQTT token，以及资源与results.txt一致。未在本机实际打开Windows XCompute，尚不能确认GUI解码或精确Source时间轴区间；请反馈无法打开或缺失wave的情况。

## 每份文件定位

### ID76 current_best

- MNKG：{'M': 1, 'N': 2048, 'K': 768, 'G': 4}；实际CK实例：5008；Native算法：None。
- 文件：`captures/id76_current_best_sqtt_02bd5874/95566_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch `18`；XCD0、SE0、active-CU-index0。
- kernel：`void ck_tile::kentry<256, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallMK32<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMK32Geometry<128> >, ck_tile::V7SmallMK32Geometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源：{'grid_work_items': [16384, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 64, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 116, 'sgpr': 64}；SQTT原始token 66368字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 163处、barrier 51处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0x2130', 'end_pc': '0x6590', 'static_mmac': 736, 'line_start': 18, 'line_end': 2992}, {'start_pc': '0x2ae0', 'end_pc': '0x37f8', 'static_mmac': 192, 'line_start': 521, 'line_end': 1056}, {'start_pc': '0x2ab0', 'end_pc': '0x37f4', 'static_mmac': 192, 'line_start': 515, 'line_end': 1055}, {'start_pc': '0x459c', 'end_pc': '0x52e8', 'static_mmac': 192, 'line_start': 1654, 'line_end': 2196}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`buffer_load_dwordx4 v[105:106], s[4:7], s15 idxen offen slc lds// 000000002B04: E05F3000 0F010069`
  指令定位：`buffer_load_dwordx4 v[107:108], s[8:11], s14 idxen offen lds// 000000002B14: E05D3000 0E02006B`
  指令定位：`buffer_load_dwordx4 v[107:108], s[8:11], s14 idxen offen lds// 000000002B24: E05D3000 0E02006B`
  指令定位：`s_waitcnt lgkmcnt(6)                                       // 000000002B2C: BF8CC67F`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[17:20], v[77:78], v[53:54], v[17:20]  // 000000002B30: FFD40011 04466B4D`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[41:44], v[77:78], v[55:56], v[41:44]  // 000000002B3C: FFD40029 04A66F4D`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[33:36], v[77:78], v[49:50], v[33:36]  // 000000002B44: FFD40021 0486634D`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[29:32], v[77:78], v[51:52], v[29:32]  // 000000002B4C: FFD4001D 0476674D`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[13:16], v[73:74], v[53:54], v[13:16]  // 000000002B54: FFD4000D 04366B49`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[9:12], v[73:74], v[55:56], v[9:12]  // 000000002B5C: FFD40009 04266F49`

### ID76 frozen_native

- MNKG：{'M': 1, 'N': 2048, 'K': 768, 'G': 4}；实际CK实例：None；Native算法：11285。
- 文件：`captures/id76_frozen_native_sqtt_352a066a/95597_python3.10.perf`；dispatch `30`；XCD0、SE0、active-CU-index0。
- kernel：`Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_GSU1_K1_MIWT2_2`。
- 资源：{'grid_work_items': [32768, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 128, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 104, 'sgpr': 112}；SQTT原始token 121312字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 199处、barrier 32处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0xeccd860', 'end_pc': '0xeccdb94', 'static_mmac': 32, 'line_start': 624, 'line_end': 767}, {'start_pc': '0xecced70', 'end_pc': '0xeccee0c', 'static_mmac': 8, 'line_start': 1600, 'line_end': 1625}, {'start_pc': '0xecce284', 'end_pc': '0xecce35c', 'static_mmac': 8, 'line_start': 1039, 'line_end': 1069}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`s_waitcnt lgkmcnt(4)                                       // 00000ECCD8C8: BF8CC47F`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[0:3], v[16:17], v[24:25], v[0:3]  // 00000ECCD8CC: FFD40000 04023110`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[4:7], v[18:19], v[24:25], v[4:7]  // 00000ECCD8D8: FFD40004 04123112`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[8:11], v[16:17], v[28:29], v[8:11]  // 00000ECCD8E0: FFD40008 04223910`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[12:15], v[18:19], v[28:29], v[12:15]  // 00000ECCD8E8: FFD4000C 04323912`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[0:3], v[20:21], v[26:27], v[0:3]  // 00000ECCD8F0: FFD40000 04023514`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[4:7], v[22:23], v[26:27], v[4:7]  // 00000ECCD8F8: FFD40004 04123516`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[8:11], v[20:21], v[30:31], v[8:11]  // 00000ECCD900: FFD40008 04223D14`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[12:15], v[22:23], v[30:31], v[12:15]  // 00000ECCD908: FFD4000C 04323D16`
  指令定位：`s_waitcnt vmcnt(4)                                         // 00000ECCD914: BF8C0F74`

### ID621 current_best

- MNKG：{'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实际CK实例：5003；Native算法：None。
- 文件：`captures/id621_current_best_sqtt_039fec8b/95953_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch `18`；XCD0、SE0、active-CU-index0。
- kernel：`void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源：{'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}；SQTT原始token 638304字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 163处、barrier 51处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0x2130', 'end_pc': '0x53dc', 'static_mmac': 368, 'line_start': 18, 'line_end': 2386}, {'start_pc': '0x2a70', 'end_pc': '0x3310', 'static_mmac': 96, 'line_start': 512, 'line_end': 893}, {'start_pc': '0x2a50', 'end_pc': '0x330c', 'static_mmac': 96, 'line_start': 508, 'line_end': 892}, {'start_pc': '0x3c94', 'end_pc': '0x4558', 'static_mmac': 96, 'line_start': 1348, 'line_end': 1734}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`buffer_load_dwordx4 v[57:58], s[4:7], s13 idxen offen slc lds// 000000002A94: E05F3000 0D010039`
  指令定位：`buffer_load_dwordx4 v[59:60], s[8:11], s12 idxen offen lds // 000000002AA4: E05D3000 0C02003B`
  指令定位：`s_waitcnt lgkmcnt(4)                                       // 000000002AAC: BF8CC47F`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[21:24], v[67:68], v[45:46], v[21:24]  // 000000002AB0: FFD40015 04565B43`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[37:40], v[67:68], v[47:48], v[37:40]  // 000000002ABC: FFD40025 04965F43`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[25:28], v[67:68], v[41:42], v[25:28]  // 000000002AC4: FFD40019 04665343`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[17:20], v[67:68], v[43:44], v[17:20]  // 000000002ACC: FFD40011 04465743`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[13:16], v[65:66], v[45:46], v[13:16]  // 000000002AD4: FFD4000D 04365B41`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[9:12], v[65:66], v[47:48], v[9:12]  // 000000002ADC: FFD40009 04265F41`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[5:8], v[65:66], v[41:42], v[5:8]  // 000000002AE4: FFD40005 04165341`

### ID621 frozen_native

- MNKG：{'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实际CK实例：None；Native算法：313。
- 文件：`captures/id621_frozen_native_sqtt_6b351944/95996_python3.10.perf`；dispatch `72`；XCD0、SE0、active-CU-index0。
- kernel：`Cijk_Ailk_Bljk_BBS_BH_GG_AS_UserArgs_MT128x128x16_MI16x16x1_SN_AFC1_DTL1_EPS1_GRVWA8_GRVWB8_GSU1_GSUAMB_K1_LRVW4_MIWT8_2_NLCA1_NLCB1_OBAFM1_OLDSR0_ONLL1_PGR4_PLR1_SU0_SUS0_TLDS1_ULDSWM1_USFGRO0_USSB1_WSS0_WG16_16_1_WGM1`。
- 资源：{'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 256, 'sgpr': 96}；SQTT原始token 749312字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 627处、barrier 18处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0x25d3798', 'end_pc': '0x25d3bfc', 'static_mmac': 64, 'line_start': 683, 'line_end': 862}, {'start_pc': '0x25d5b60', 'end_pc': '0x25d5c38', 'static_mmac': 16, 'line_start': 2187, 'line_end': 2218}, {'start_pc': '0x25d47a8', 'end_pc': '0x25d48cc', 'static_mmac': 16, 'line_start': 1279, 'line_end': 1318}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`s_waitcnt lgkmcnt(6)                                       // 0000025D3810: BF8CC67F`
  指令定位：`s_waitcnt vmcnt(4)                                         // 0000025D3814: BF8C0F74`
  指令定位：`s_barrier                                                  // 0000025D3818: BF8A0000`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[0:3], v[64:65], v[96:97], v[0:3]  // 0000025D3820: FFD40000 0402C140`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[4:7], v[66:67], v[96:97], v[4:7]  // 0000025D3828: FFD40004 0412C142`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[8:11], v[68:69], v[96:97], v[8:11]  // 0000025D3830: FFD40008 0422C144`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[12:15], v[70:71], v[96:97], v[12:15]  // 0000025D3838: FFD4000C 0432C146`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[16:19], v[72:73], v[96:97], v[16:19]  // 0000025D3840: FFD40010 0442C148`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[20:23], v[74:75], v[96:97], v[20:23]  // 0000025D3848: FFD40014 0452C14A`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[24:27], v[76:77], v[96:97], v[24:27]  // 0000025D3850: FFD40018 0462C14C`

### ID805 current_best

- MNKG：{'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实际CK实例：5000；Native算法：None。
- 文件：`captures/id805_current_best_sqtt_b9ff83af/96359_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch `18`；XCD0、SE0、active-CU-index0。
- kernel：`void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源：{'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}；SQTT原始token 2802496字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 121处、barrier 51处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0x2130', 'end_pc': '0x8fb4', 'static_mmac': 1472, 'line_start': 18, 'line_end': 4552}, {'start_pc': '0x2c6c', 'end_pc': '0x3f3c', 'static_mmac': 384, 'line_start': 616, 'line_end': 1317}, {'start_pc': '0x2c2c', 'end_pc': '0x3f38', 'static_mmac': 384, 'line_start': 608, 'line_end': 1316}, {'start_pc': '0x5474', 'end_pc': '0x6788', 'static_mmac': 384, 'line_start': 2213, 'line_end': 2923}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen slc lds// 000000002C90: E05F3000 0D0100B1`
  指令定位：`buffer_load_dwordx4 v206, s[8:11], s12 offen lds           // 000000002CA0: E05D1000 0C0200CE`
  指令定位：`s_waitcnt lgkmcnt(8)                                       // 000000002CA8: BF8CC87F`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[109:112], v[193:194], v[157:158], v[109:112]  // 000000002CAC: FFD4006D 05B73BC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[141:144], v[193:194], v[159:160], v[141:144]  // 000000002CB8: FFD4008D 06373FC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[137:140], v[193:194], v[153:154], v[137:140]  // 000000002CC0: FFD40089 062733C1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[129:132], v[193:194], v[155:156], v[129:132]  // 000000002CC8: FFD40081 060737C1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[113:116], v[193:194], v[149:150], v[113:116]  // 000000002CD0: FFD40071 05C72BC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[105:108], v[193:194], v[151:152], v[105:108]  // 000000002CD8: FFD40069 05A72FC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[101:104], v[193:194], v[145:146], v[101:104]  // 000000002CE0: FFD40065 059723C1`

### ID805 frozen_native

- MNKG：{'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实际CK实例：None；Native算法：11279。
- 文件：`captures/id805_frozen_native_sqtt_3b6b79db/96419_python3.10.perf`；dispatch `219`；XCD0、SE0、active-CU-index0。
- kernel：`Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x64x32_MI16x16x1_SN_GSU1_K1_MIWT4_2_WGM8`。
- 资源：{'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 188, 'sgpr': 112}；SQTT原始token 1582080字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 333处、barrier 35处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0xec5c4c0', 'end_pc': '0xec5c974', 'static_mmac': 64, 'line_start': 691, 'line_end': 882}, {'start_pc': '0xec5c97c', 'end_pc': '0xec5ce30', 'static_mmac': 64, 'line_start': 886, 'line_end': 1077}, {'start_pc': '0xec5e978', 'end_pc': '0xec5ea5c', 'static_mmac': 16, 'line_start': 2246, 'line_end': 2279}, {'start_pc': '0xec5da20', 'end_pc': '0xec5db48', 'static_mmac': 16, 'line_start': 1509, 'line_end': 1549}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`s_waitcnt lgkmcnt(6)                                       // 00000EC5C548: BF8CC67F`
  指令定位：`s_waitcnt vmcnt(6)                                         // 00000EC5C54C: BF8C0F76`
  指令定位：`s_barrier                                                  // 00000EC5C550: BF8A0000`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[0:3], v[64:65], v[80:81], v[0:3]  // 00000EC5C554: FFD40000 0402A140`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[4:7], v[66:67], v[80:81], v[4:7]  // 00000EC5C560: FFD40004 0412A142`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[8:11], v[68:69], v[80:81], v[8:11]  // 00000EC5C568: FFD40008 0422A144`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[12:15], v[70:71], v[80:81], v[12:15]  // 00000EC5C570: FFD4000C 0432A146`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[16:19], v[64:65], v[84:85], v[16:19]  // 00000EC5C578: FFD40010 0442A940`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[20:23], v[66:67], v[84:85], v[20:23]  // 00000EC5C580: FFD40014 0452A942`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[24:27], v[68:69], v[84:85], v[24:27]  // 00000EC5C588: FFD40018 0462A944`

### ID1072 current_best

- MNKG：{'M': 2, 'N': 4096, 'K': 7168, 'G': 16}；实际CK实例：5006；Native算法：None。
- 文件：`captures/id1072_current_best_sqtt_869739e0/96784_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch `18`；XCD0、SE0、active-CU-index0。
- kernel：`void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7CacheNoSlc<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源：{'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}；SQTT原始token 4780288字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 121处、barrier 51处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0x2130', 'end_pc': '0x9068', 'static_mmac': 1472, 'line_start': 18, 'line_end': 4595}, {'start_pc': '0x2d20', 'end_pc': '0x3ff0', 'static_mmac': 384, 'line_start': 659, 'line_end': 1360}, {'start_pc': '0x2ce0', 'end_pc': '0x3fec', 'static_mmac': 384, 'line_start': 651, 'line_end': 1359}, {'start_pc': '0x5528', 'end_pc': '0x683c', 'static_mmac': 384, 'line_start': 2256, 'line_end': 2966}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen lds// 000000002D44: E05D3000 0D0100B1`
  指令定位：`buffer_load_dwordx4 v206, s[8:11], s12 offen lds           // 000000002D54: E05D1000 0C0200CE`
  指令定位：`s_waitcnt lgkmcnt(8)                                       // 000000002D5C: BF8CC87F`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[109:112], v[193:194], v[157:158], v[109:112]  // 000000002D60: FFD4006D 05B73BC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[141:144], v[193:194], v[159:160], v[141:144]  // 000000002D6C: FFD4008D 06373FC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[137:140], v[193:194], v[153:154], v[137:140]  // 000000002D74: FFD40089 062733C1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[129:132], v[193:194], v[155:156], v[129:132]  // 000000002D7C: FFD40081 060737C1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[113:116], v[193:194], v[149:150], v[113:116]  // 000000002D84: FFD40071 05C72BC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[105:108], v[193:194], v[151:152], v[105:108]  // 000000002D8C: FFD40069 05A72FC1`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[101:104], v[193:194], v[145:146], v[101:104]  // 000000002D94: FFD40065 059723C1`

### ID1072 frozen_native

- MNKG：{'M': 2, 'N': 4096, 'K': 7168, 'G': 16}；实际CK实例：None；Native算法：11278。
- 文件：`captures/id1072_frozen_native_sqtt_94b604a7/96851_python3.10.perf`；dispatch `360`；XCD0、SE0、active-CU-index0。
- kernel：`Custom_Cijk_Ailk_Bljk_BBS_BH_Bias_GG_AS_SAV_UserArgs_MT128x64x32_MI16x16x1_SN_GSU1_K1_MIWT4_2_WGM1`。
- 资源：{'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 188, 'sgpr': 112}；SQTT原始token 2976480字节。
- Source搜索：`buffer_load`、`ds_read`、`s_waitcnt`、`s_barrier`、`v_mmac`。本内核静态wait 333处、barrier 35处；静态数量不等于动态执行次数。
- 需要检验：CTA数量与资源是否限制并行；A/B搬运和MMAC是否重叠；共享LDS与同步的开销是否已被计算覆盖。HBM总流量不能独自判定重复的是B。

- 包含MMAC的静态后向分支区间：`[{'start_pc': '0xec53d34', 'end_pc': '0xec541e8', 'static_mmac': 64, 'line_start': 659, 'line_end': 850}, {'start_pc': '0xec541f0', 'end_pc': '0xec546a4', 'static_mmac': 64, 'line_start': 854, 'line_end': 1045}, {'start_pc': '0xec561ec', 'end_pc': '0xec562d4', 'static_mmac': 16, 'line_start': 2214, 'line_end': 2248}, {'start_pc': '0xec55294', 'end_pc': '0xec553bc', 'static_mmac': 16, 'line_start': 1477, 'line_end': 1517}]`。优先在这些PC区间查看高Hit Count主循环，避免把开头参数s_load等待当作B流水瓶颈。
  指令定位：`s_waitcnt lgkmcnt(6)                                       // 00000EC53DBC: BF8CC67F`
  指令定位：`s_waitcnt vmcnt(6)                                         // 00000EC53DC0: BF8C0F76`
  指令定位：`s_barrier                                                  // 00000EC53DC4: BF8A0000`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[0:3], v[64:65], v[80:81], v[0:3]  // 00000EC53DC8: FFD40000 0402A140`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[4:7], v[66:67], v[80:81], v[4:7]  // 00000EC53DD4: FFD40004 0412A142`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[8:11], v[68:69], v[80:81], v[8:11]  // 00000EC53DDC: FFD40008 0422A144`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[12:15], v[70:71], v[80:81], v[12:15]  // 00000EC53DE4: FFD4000C 0432A146`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[16:19], v[64:65], v[84:85], v[16:19]  // 00000EC53DEC: FFD40010 0442A940`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[20:23], v[66:67], v[84:85], v[20:23]  // 00000EC53DF4: FFD40014 0452A942`
  指令定位：`v_mmac_f32_16x16x16_bf16 v[24:27], v[68:69], v[84:85], v[24:27]  // 00000EC53DFC: FFD40018 0462A944`
