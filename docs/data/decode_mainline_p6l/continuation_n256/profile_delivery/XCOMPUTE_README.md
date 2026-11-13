# N256双槽/三槽与当前CK及深度控制对照

全部55个预登记发现配对结束后采集；固定621/805/1009/312并补齐实际存在的收益/平坦/回退类别，另固定ID805的D2/D3直接控制。主对照base是当前最佳CK；depth_control的base是N256双槽6030，candidate是三槽6031，不能混称为当前最佳CK。这些诊断样本均不是最终采用结论。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
重点核对D2的vmcnt(0)与D3保留后续请求的vmcnt(2/3)，并检查未来LDS读取与当前MMAC是否重叠、LG0是否在下一barrier之前；结合实际VGPR/LDS/驻留和访存计数。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。
本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID621 id621_d3 base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3n256_id621_base_cu0_3a3131de/768022_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+1.608%；平均驻留wave/CU 6.3491794127498045；HBM读取 154850816字节；wait/wavecycle 0.14917933772860983。
- 主循环PC `[{'start_pc': '0x2a70', 'end_pc': '0x3310', 'static_mmac': 96, 'line_start': 512, 'line_end': 893}, {'start_pc': '0x2a50', 'end_pc': '0x330c', 'static_mmac': 96, 'line_start': 508, 'line_end': 892}, {'start_pc': '0x3c94', 'end_pc': '0x4558', 'static_mmac': 96, 'line_start': 1348, 'line_end': 1734}, {'start_pc': '0x33b0', 'end_pc': '0x3a78', 'static_mmac': 72, 'line_start': 933, 'line_end': 1241}]`。

  指令PC：`buffer_load_dwordx4 v[57:58], s[4:7], s13 idxen offen slc lds// 000000002A94: E05F3000 0D010039`
  指令PC：`buffer_load_dwordx4 v[59:60], s[8:11], s12 idxen offen lds // 000000002AA4: E05D3000 0C02003B`
  指令PC：`s_waitcnt lgkmcnt(4)                                       // 000000002AAC: BF8CC47F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[67:68], v[45:46], v[21:24]  // 000000002AB0: FFD40015 04565B43`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[37:40], v[67:68], v[47:48], v[37:40]  // 000000002ABC: FFD40025 04965F43`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[67:68], v[41:42], v[25:28]  // 000000002AC4: FFD40019 04665343`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[67:68], v[43:44], v[17:20]  // 000000002ACC: FFD40011 04465743`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[65:66], v[45:46], v[13:16]  // 000000002AD4: FFD4000D 04365B41`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[65:66], v[47:48], v[9:12]  // 000000002ADC: FFD40009 04265F41`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[65:66], v[41:42], v[5:8]  // 000000002AE4: FFD40005 04165341`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[65:66], v[43:44], v[1:4]  // 000000002AEC: FFD40001 04065741`
  指令PC：`s_waitcnt vmcnt(4)                                         // 000000002AF8: BF8C0F74`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000002AFC: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000002B00: BF8A0000`
  指令PC：`ds_read_m32x16_b16 v[45:48], v78 offset:24576              // 000000002B10: D9F66000 2D00004E`
  指令PC：`ds_read_m32x16_b16 v[41:44], v79 offset:24576              // 000000002B18: D9F66000 2900004F`
## ID621 id621_d3 candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id621_candidate_cu0_3680ae2a/768056_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<4, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 128, 'lds_bytes': 61440, 'scratch_bytes': 0, 'vgpr': 184, 'sgpr': 64}。
- 正式candidate/base变化+1.608%；平均驻留wave/CU 6.677488097532424；HBM读取 152155904字节；wait/wavecycle 0.1526330276633463。
- 主循环PC `[{'start_pc': '0x5b394', 'end_pc': '0x5be70', 'line_start': 327, 'line_end': 754}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[141:142], v[125:126], v[25:28]  // 00000005B394: FFD40019 0466FB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[141:142], v[127:128], v[29:32]  // 00000005B39C: FFD4001D 0476FF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[141:142], v[121:122], v[21:24]  // 00000005B3A4: FFD40015 0456F38D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[141:142], v[123:124], v[17:20]  // 00000005B3AC: FFD40011 0446F78D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[141:142], v[117:118], v[13:16]  // 00000005B3B4: FFD4000D 0436EB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[141:142], v[119:120], v[9:12]  // 00000005B3BC: FFD40009 0426EF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[141:142], v[109:110], v[5:8]  // 00000005B3C4: FFD40005 0416DB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[141:142], v[111:112], v[1:4]  // 00000005B3CC: FFD40001 0406DF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[139:140], v[113:114], v[25:28]  // 00000005B3D4: FFD40019 0466E38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[139:140], v[115:116], v[29:32]  // 00000005B3DC: FFD4001D 0476E78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[139:140], v[105:106], v[21:24]  // 00000005B3E4: FFD40015 0456D38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[139:140], v[107:108], v[17:20]  // 00000005B3EC: FFD40011 0446D78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[139:140], v[101:102], v[13:16]  // 00000005B3F4: FFD4000D 0436CB8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[139:140], v[103:104], v[9:12]  // 00000005B3FC: FFD40009 0426CF8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[139:140], v[97:98], v[5:8]  // 00000005B404: FFD40005 0416C38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[139:140], v[99:100], v[1:4]  // 00000005B40C: FFD40001 0406C78B`
## ID805 id805_d3 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3n256_id805_base_cu0_f1b07bc7/768096_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+1.982%；平均驻留wave/CU 6.681667498133796；HBM读取 566660736字节；wait/wavecycle 0.17416140993529075。
- 主循环PC `[{'start_pc': '0x2c6c', 'end_pc': '0x3f3c', 'static_mmac': 384, 'line_start': 616, 'line_end': 1317}, {'start_pc': '0x2c2c', 'end_pc': '0x3f38', 'static_mmac': 384, 'line_start': 608, 'line_end': 1316}, {'start_pc': '0x5474', 'end_pc': '0x6788', 'static_mmac': 384, 'line_start': 2213, 'line_end': 2923}, {'start_pc': '0x409c', 'end_pc': '0x4f20', 'static_mmac': 288, 'line_start': 1405, 'line_end': 1956}]`。

  指令PC：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen slc lds// 000000002C90: E05F3000 0D0100B1`
  指令PC：`buffer_load_dwordx4 v206, s[8:11], s12 offen lds           // 000000002CA0: E05D1000 0C0200CE`
  指令PC：`s_waitcnt lgkmcnt(8)                                       // 000000002CA8: BF8CC87F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[109:112], v[193:194], v[157:158], v[109:112]  // 000000002CAC: FFD4006D 05B73BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[141:144], v[193:194], v[159:160], v[141:144]  // 000000002CB8: FFD4008D 06373FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[137:140], v[193:194], v[153:154], v[137:140]  // 000000002CC0: FFD40089 062733C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[129:132], v[193:194], v[155:156], v[129:132]  // 000000002CC8: FFD40081 060737C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[113:116], v[193:194], v[149:150], v[113:116]  // 000000002CD0: FFD40071 05C72BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[105:108], v[193:194], v[151:152], v[105:108]  // 000000002CD8: FFD40069 05A72FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[101:104], v[193:194], v[145:146], v[101:104]  // 000000002CE0: FFD40065 059723C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[97:100], v[193:194], v[147:148], v[97:100]  // 000000002CE8: FFD40061 058727C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[93:96], v[191:192], v[157:158], v[93:96]  // 000000002CF0: FFD4005D 05773BBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[89:92], v[191:192], v[159:160], v[89:92]  // 000000002CF8: FFD40059 05673FBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[85:88], v[191:192], v[153:154], v[85:88]  // 000000002D00: FFD40055 055733BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[81:84], v[191:192], v[155:156], v[81:84]  // 000000002D08: FFD40051 054737BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[77:80], v[191:192], v[149:150], v[77:80]  // 000000002D10: FFD4004D 05372BBF`
## ID805 id805_d3 candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id805_candidate_cu0_2b67776b/768150_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<4, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 61440, 'scratch_bytes': 0, 'vgpr': 184, 'sgpr': 64}。
- 正式candidate/base变化+1.982%；平均驻留wave/CU 6.971677102381574；HBM读取 570430208字节；wait/wavecycle 0.165645571108691。
- 主循环PC `[{'start_pc': '0x5b394', 'end_pc': '0x5be70', 'line_start': 327, 'line_end': 754}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[141:142], v[125:126], v[25:28]  // 00000005B394: FFD40019 0466FB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[141:142], v[127:128], v[29:32]  // 00000005B39C: FFD4001D 0476FF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[141:142], v[121:122], v[21:24]  // 00000005B3A4: FFD40015 0456F38D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[141:142], v[123:124], v[17:20]  // 00000005B3AC: FFD40011 0446F78D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[141:142], v[117:118], v[13:16]  // 00000005B3B4: FFD4000D 0436EB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[141:142], v[119:120], v[9:12]  // 00000005B3BC: FFD40009 0426EF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[141:142], v[109:110], v[5:8]  // 00000005B3C4: FFD40005 0416DB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[141:142], v[111:112], v[1:4]  // 00000005B3CC: FFD40001 0406DF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[139:140], v[113:114], v[25:28]  // 00000005B3D4: FFD40019 0466E38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[139:140], v[115:116], v[29:32]  // 00000005B3DC: FFD4001D 0476E78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[139:140], v[105:106], v[21:24]  // 00000005B3E4: FFD40015 0456D38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[139:140], v[107:108], v[17:20]  // 00000005B3EC: FFD40011 0446D78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[139:140], v[101:102], v[13:16]  // 00000005B3F4: FFD4000D 0436CB8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[139:140], v[103:104], v[9:12]  // 00000005B3FC: FFD40009 0426CF8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[139:140], v[97:98], v[5:8]  // 00000005B404: FFD40005 0416C38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[139:140], v[99:100], v[1:4]  // 00000005B40C: FFD40001 0406C78B`
## ID1009 id1009_d3 base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例5002；GPU4/epoch0。
- 文件 `captures/h3n256_id1009_base_cu0_3238a67b/768197_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-8.456%；平均驻留wave/CU 6.520838957332326；HBM读取 807639296字节；wait/wavecycle 0.19628440818165463。
- 主循环PC `[{'start_pc': '0x2d20', 'end_pc': '0x3ff0', 'static_mmac': 384, 'line_start': 659, 'line_end': 1360}, {'start_pc': '0x2ce0', 'end_pc': '0x3fec', 'static_mmac': 384, 'line_start': 651, 'line_end': 1359}, {'start_pc': '0x5528', 'end_pc': '0x683c', 'static_mmac': 384, 'line_start': 2256, 'line_end': 2966}, {'start_pc': '0x4150', 'end_pc': '0x4fd4', 'static_mmac': 288, 'line_start': 1448, 'line_end': 1999}]`。

  指令PC：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen slc lds// 000000002D44: E05F3000 0D0100B1`
  指令PC：`buffer_load_dwordx4 v206, s[8:11], s12 offen lds           // 000000002D54: E05D1000 0C0200CE`
  指令PC：`s_waitcnt lgkmcnt(8)                                       // 000000002D5C: BF8CC87F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[109:112], v[193:194], v[157:158], v[109:112]  // 000000002D60: FFD4006D 05B73BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[141:144], v[193:194], v[159:160], v[141:144]  // 000000002D6C: FFD4008D 06373FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[137:140], v[193:194], v[153:154], v[137:140]  // 000000002D74: FFD40089 062733C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[129:132], v[193:194], v[155:156], v[129:132]  // 000000002D7C: FFD40081 060737C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[113:116], v[193:194], v[149:150], v[113:116]  // 000000002D84: FFD40071 05C72BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[105:108], v[193:194], v[151:152], v[105:108]  // 000000002D8C: FFD40069 05A72FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[101:104], v[193:194], v[145:146], v[101:104]  // 000000002D94: FFD40065 059723C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[97:100], v[193:194], v[147:148], v[97:100]  // 000000002D9C: FFD40061 058727C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[93:96], v[191:192], v[157:158], v[93:96]  // 000000002DA4: FFD4005D 05773BBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[89:92], v[191:192], v[159:160], v[89:92]  // 000000002DAC: FFD40059 05673FBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[85:88], v[191:192], v[153:154], v[85:88]  // 000000002DB4: FFD40055 055733BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[81:84], v[191:192], v[155:156], v[81:84]  // 000000002DBC: FFD40051 054737BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[77:80], v[191:192], v[149:150], v[77:80]  // 000000002DC4: FFD4004D 05372BBF`
## ID1009 id1009_d3 candidate

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id1009_candidate_cu0_42939274/768264_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<1, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 52224, 'scratch_bytes': 0, 'vgpr': 68, 'sgpr': 64}。
- 正式candidate/base变化-8.456%；平均驻留wave/CU 7.183829738885909；HBM读取 805665472字节；wait/wavecycle 0.061065478857510395。
- 主循环PC `[{'start_pc': '0x1c328', 'end_pc': '0x1c8f8', 'line_start': 257, 'line_end': 523}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[43:44], v[25:26], v[1:4]  // 00000001C328: FFD40001 0406332B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[43:44], v[27:28], v[5:8]  // 00000001C334: FFD40005 0416372B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[45:46], v[29:30], v[1:4]  // 00000001C33C: FFD40001 04063B2D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[45:46], v[31:32], v[5:8]  // 00000001C344: FFD40005 04163F2D`
  指令PC：`s_waitcnt vmcnt(2)                                         // 00000001C358: BF8C0F72`
  指令PC：`s_waitcnt vmcnt(3)                                         // 00000001C368: BF8C0F73`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000001C36C: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000001C370: BF8A0000`
  指令PC：`ds_read_b64 v[39:40], v64                                  // 00000001C374: D8EC0000 27000040`
  指令PC：`ds_read_m32x16_b16 v[25:28], v65                           // 00000001C37C: D9F60000 19000041`
  指令PC：`ds_read_b64 v[31:32], v66                                  // 00000001C384: D8EC0000 1F000042`
  指令PC：`ds_read_m32x16_b16 v[17:20], v67                           // 00000001C390: D9F60000 11000043`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s2 idxen offen slc lds// 00000001C3A8: E05F3000 02040021`
  指令PC：`buffer_load_dwordx4 v59, s[12:15], s2 offen lds            // 00000001C3B8: E05D1000 0203003B`
  指令PC：`buffer_load_dwordx4 v59, s[12:15], s2 offen lds            // 00000001C3C8: E05D1000 0203003B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 00000001C3D0: FFD40001 04061B25`
## ID312 id312_d3 base

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例5000；GPU4/epoch0。
- 文件 `captures/h3n256_id312_base_cu0_2d5a53ff/768324_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+0.160%；平均驻留wave/CU 7.719593047275043；HBM读取 1091714880字节；wait/wavecycle 0.20591625481951603。
- 主循环PC `[{'start_pc': '0x2c6c', 'end_pc': '0x3f3c', 'static_mmac': 384, 'line_start': 616, 'line_end': 1317}, {'start_pc': '0x2c2c', 'end_pc': '0x3f38', 'static_mmac': 384, 'line_start': 608, 'line_end': 1316}, {'start_pc': '0x5474', 'end_pc': '0x6788', 'static_mmac': 384, 'line_start': 2213, 'line_end': 2923}, {'start_pc': '0x409c', 'end_pc': '0x4f20', 'static_mmac': 288, 'line_start': 1405, 'line_end': 1956}]`。

  指令PC：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen slc lds// 000000002C90: E05F3000 0D0100B1`
  指令PC：`buffer_load_dwordx4 v206, s[8:11], s12 offen lds           // 000000002CA0: E05D1000 0C0200CE`
  指令PC：`s_waitcnt lgkmcnt(8)                                       // 000000002CA8: BF8CC87F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[109:112], v[193:194], v[157:158], v[109:112]  // 000000002CAC: FFD4006D 05B73BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[141:144], v[193:194], v[159:160], v[141:144]  // 000000002CB8: FFD4008D 06373FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[137:140], v[193:194], v[153:154], v[137:140]  // 000000002CC0: FFD40089 062733C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[129:132], v[193:194], v[155:156], v[129:132]  // 000000002CC8: FFD40081 060737C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[113:116], v[193:194], v[149:150], v[113:116]  // 000000002CD0: FFD40071 05C72BC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[105:108], v[193:194], v[151:152], v[105:108]  // 000000002CD8: FFD40069 05A72FC1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[101:104], v[193:194], v[145:146], v[101:104]  // 000000002CE0: FFD40065 059723C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[97:100], v[193:194], v[147:148], v[97:100]  // 000000002CE8: FFD40061 058727C1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[93:96], v[191:192], v[157:158], v[93:96]  // 000000002CF0: FFD4005D 05773BBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[89:92], v[191:192], v[159:160], v[89:92]  // 000000002CF8: FFD40059 05673FBF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[85:88], v[191:192], v[153:154], v[85:88]  // 000000002D00: FFD40055 055733BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[81:84], v[191:192], v[155:156], v[81:84]  // 000000002D08: FFD40051 054737BF`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[77:80], v[191:192], v[149:150], v[77:80]  // 000000002D10: FFD4004D 05372BBF`
## ID312 id312_d3 candidate

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id312_candidate_cu0_2e7b84a4/768399_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<2, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 55296, 'scratch_bytes': 0, 'vgpr': 108, 'sgpr': 64}。
- 正式candidate/base变化+0.160%；平均驻留wave/CU 7.682568602939937；HBM读取 1090472768字节；wait/wavecycle 0.09375579832862906。
- 主循环PC `[{'start_pc': '0x380ac', 'end_pc': '0x38828', 'line_start': 284, 'line_end': 603}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[77:78], v[61:62], v[9:12]  // 0000000380AC: FFD40009 04267B4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[77:78], v[63:64], v[13:16]  // 0000000380B4: FFD4000D 04367F4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[77:78], v[53:54], v[5:8]  // 0000000380BC: FFD40005 04166B4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[77:78], v[55:56], v[1:4]  // 0000000380C4: FFD40001 04066F4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[75:76], v[57:58], v[9:12]  // 0000000380CC: FFD40009 0426734B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[75:76], v[59:60], v[13:16]  // 0000000380D4: FFD4000D 0436774B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[75:76], v[49:50], v[5:8]  // 0000000380DC: FFD40005 0416634B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[75:76], v[51:52], v[1:4]  // 0000000380E4: FFD40001 0406674B`
  指令PC：`s_waitcnt vmcnt(2)                                         // 0000000380F8: BF8C0F72`
  指令PC：`s_waitcnt vmcnt(3)                                         // 000000038108: BF8C0F73`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000003810C: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000038110: BF8A0000`
  指令PC：`ds_read_b64 v[63:64], v100                                 // 000000038114: D8EC0000 3F000064`
  指令PC：`ds_read_m32x16_b16 v[53:56], v101                          // 00000003811C: D9F60000 35000065`
  指令PC：`ds_read_m32x16_b16 v[45:48], v102                          // 000000038124: D9F60000 2D000066`
  指令PC：`ds_read_b64 v[59:60], v103                                 // 00000003812C: D8EC0000 3B000067`
## ID633 id633_d3 base

- MNKG {'M': 4, 'N': 1024, 'K': 2048, 'G': 8}；实例5008；GPU4/epoch0。
- 文件 `captures/h3n256_id633_base_cu1_a7510df1/768493_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index1。
- kernel `void ck_tile::kentry<256, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallMK32<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMK32Geometry<128> >, ck_tile::V7SmallMK32Geometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [16384, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 64, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 116, 'sgpr': 64}。
- 正式candidate/base变化+11.489%；平均驻留wave/CU 2.2014401261123226；HBM读取 34263104字节；wait/wavecycle 0.059274097182056916。
- 主循环PC `[{'start_pc': '0x2ae0', 'end_pc': '0x37f8', 'static_mmac': 192, 'line_start': 521, 'line_end': 1056}, {'start_pc': '0x2ab0', 'end_pc': '0x37f4', 'static_mmac': 192, 'line_start': 515, 'line_end': 1055}, {'start_pc': '0x459c', 'end_pc': '0x52e8', 'static_mmac': 192, 'line_start': 1654, 'line_end': 2196}, {'start_pc': '0x3898', 'end_pc': '0x42c0', 'static_mmac': 144, 'line_start': 1096, 'line_end': 1521}]`。

  指令PC：`buffer_load_dwordx4 v[105:106], s[4:7], s15 idxen offen slc lds// 000000002B04: E05F3000 0F010069`
  指令PC：`buffer_load_dwordx4 v[107:108], s[8:11], s14 idxen offen lds// 000000002B14: E05D3000 0E02006B`
  指令PC：`buffer_load_dwordx4 v[107:108], s[8:11], s14 idxen offen lds// 000000002B24: E05D3000 0E02006B`
  指令PC：`s_waitcnt lgkmcnt(6)                                       // 000000002B2C: BF8CC67F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[77:78], v[53:54], v[17:20]  // 000000002B30: FFD40011 04466B4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[41:44], v[77:78], v[55:56], v[41:44]  // 000000002B3C: FFD40029 04A66F4D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[33:36], v[77:78], v[49:50], v[33:36]  // 000000002B44: FFD40021 0486634D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[77:78], v[51:52], v[29:32]  // 000000002B4C: FFD4001D 0476674D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[73:74], v[53:54], v[13:16]  // 000000002B54: FFD4000D 04366B49`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[73:74], v[55:56], v[9:12]  // 000000002B5C: FFD40009 04266F49`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[73:74], v[49:50], v[5:8]  // 000000002B64: FFD40005 04166349`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[73:74], v[51:52], v[1:4]  // 000000002B6C: FFD40001 04066749`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[79:80], v[69:70], v[17:20]  // 000000002B74: FFD40011 04468B4F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[41:44], v[79:80], v[71:72], v[41:44]  // 000000002B7C: FFD40029 04A68F4F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[33:36], v[79:80], v[65:66], v[33:36]  // 000000002B84: FFD40021 0486834F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[79:80], v[67:68], v[29:32]  // 000000002B8C: FFD4001D 0476874F`
## ID633 id633_d3 candidate

- MNKG {'M': 4, 'N': 1024, 'K': 2048, 'G': 8}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id633_candidate_cu1_124478ab/768533_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index1。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<1, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [16384, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 32, 'lds_bytes': 52224, 'scratch_bytes': 0, 'vgpr': 68, 'sgpr': 64}。
- 正式candidate/base变化+11.489%；平均驻留wave/CU 2.49919426709717；HBM读取 33775744字节；wait/wavecycle 0.12941077191273023。
- 主循环PC `[{'start_pc': '0x1c328', 'end_pc': '0x1c8f8', 'line_start': 257, 'line_end': 523}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[43:44], v[25:26], v[1:4]  // 00000001C328: FFD40001 0406332B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[43:44], v[27:28], v[5:8]  // 00000001C334: FFD40005 0416372B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[45:46], v[29:30], v[1:4]  // 00000001C33C: FFD40001 04063B2D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[45:46], v[31:32], v[5:8]  // 00000001C344: FFD40005 04163F2D`
  指令PC：`s_waitcnt vmcnt(2)                                         // 00000001C358: BF8C0F72`
  指令PC：`s_waitcnt vmcnt(3)                                         // 00000001C368: BF8C0F73`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000001C36C: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000001C370: BF8A0000`
  指令PC：`ds_read_b64 v[39:40], v64                                  // 00000001C374: D8EC0000 27000040`
  指令PC：`ds_read_m32x16_b16 v[25:28], v65                           // 00000001C37C: D9F60000 19000041`
  指令PC：`ds_read_b64 v[31:32], v66                                  // 00000001C384: D8EC0000 1F000042`
  指令PC：`ds_read_m32x16_b16 v[17:20], v67                           // 00000001C390: D9F60000 11000043`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s2 idxen offen slc lds// 00000001C3A8: E05F3000 02040021`
  指令PC：`buffer_load_dwordx4 v59, s[12:15], s2 offen lds            // 00000001C3B8: E05D1000 0203003B`
  指令PC：`buffer_load_dwordx4 v59, s[12:15], s2 offen lds            // 00000001C3C8: E05D1000 0203003B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 00000001C3D0: FFD40001 04061B25`
## ID805 depth_id805_d2_d3 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6030；GPU4/epoch0。
- 文件 `captures/h3n256_id805_base_cu0_19d99f04/768567_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 168, 'sgpr': 64}。
- 正式candidate/base变化-3.912%；平均驻留wave/CU 6.668823808173332；HBM读取 571220416字节；wait/wavecycle 0.15092905129077658。
- 主循环PC `[{'start_pc': '0x59688', 'end_pc': '0x59d20', 'line_start': 282, 'line_end': 526}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[137:138], v[93:94], v[25:28]  // 000000059688: FFD40019 0466BB89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[137:138], v[95:96], v[29:32]  // 000000059690: FFD4001D 0476BF89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[137:138], v[89:90], v[21:24]  // 000000059698: FFD40015 0456B389`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[137:138], v[91:92], v[17:20]  // 0000000596A0: FFD40011 0446B789`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[137:138], v[85:86], v[13:16]  // 0000000596A8: FFD4000D 0436AB89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[137:138], v[87:88], v[9:12]  // 0000000596B0: FFD40009 0426AF89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[137:138], v[81:82], v[5:8]  // 0000000596B8: FFD40005 0416A389`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[137:138], v[83:84], v[1:4]  // 0000000596C0: FFD40001 0406A789`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[135:136], v[77:78], v[25:28]  // 0000000596C8: FFD40019 04669B87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[135:136], v[79:80], v[29:32]  // 0000000596D0: FFD4001D 04769F87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[135:136], v[73:74], v[21:24]  // 0000000596D8: FFD40015 04569387`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[135:136], v[75:76], v[17:20]  // 0000000596E0: FFD40011 04469787`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[135:136], v[69:70], v[13:16]  // 0000000596E8: FFD4000D 04368B87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[135:136], v[71:72], v[9:12]  // 0000000596F0: FFD40009 04268F87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[135:136], v[65:66], v[5:8]  // 0000000596F8: FFD40005 04168387`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[135:136], v[67:68], v[1:4]  // 000000059700: FFD40001 04068787`
## ID805 depth_id805_d2_d3 candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6031；GPU4/epoch0。
- 文件 `captures/h3n256_id805_candidate_cu0_1d7c9b09/768622_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<4, 3>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 61440, 'scratch_bytes': 0, 'vgpr': 184, 'sgpr': 64}。
- 正式candidate/base变化-3.912%；平均驻留wave/CU 7.017697134191188；HBM读取 576469504字节；wait/wavecycle 0.15255594932414965。
- 主循环PC `[{'start_pc': '0x5b394', 'end_pc': '0x5be70', 'line_start': 327, 'line_end': 754}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[141:142], v[125:126], v[25:28]  // 00000005B394: FFD40019 0466FB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[141:142], v[127:128], v[29:32]  // 00000005B39C: FFD4001D 0476FF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[141:142], v[121:122], v[21:24]  // 00000005B3A4: FFD40015 0456F38D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[141:142], v[123:124], v[17:20]  // 00000005B3AC: FFD40011 0446F78D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[141:142], v[117:118], v[13:16]  // 00000005B3B4: FFD4000D 0436EB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[141:142], v[119:120], v[9:12]  // 00000005B3BC: FFD40009 0426EF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[141:142], v[109:110], v[5:8]  // 00000005B3C4: FFD40005 0416DB8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[141:142], v[111:112], v[1:4]  // 00000005B3CC: FFD40001 0406DF8D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[139:140], v[113:114], v[25:28]  // 00000005B3D4: FFD40019 0466E38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[139:140], v[115:116], v[29:32]  // 00000005B3DC: FFD4001D 0476E78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[139:140], v[105:106], v[21:24]  // 00000005B3E4: FFD40015 0456D38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[139:140], v[107:108], v[17:20]  // 00000005B3EC: FFD40011 0446D78B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[139:140], v[101:102], v[13:16]  // 00000005B3F4: FFD4000D 0436CB8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[139:140], v[103:104], v[9:12]  // 00000005B3FC: FFD40009 0426CF8B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[139:140], v[97:98], v[5:8]  // 00000005B404: FFD40005 0416C38B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[139:140], v[99:100], v[1:4]  // 00000005B40C: FFD40001 0406C78B`
