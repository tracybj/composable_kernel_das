# 提前计算当前MMAC的阶段顺序与当前CK对照

全部46个预登记发现配对结束后采集；固定621/805/1009/312并补齐实际存在的收益/平坦/回退类别。候选选择和这些诊断样本均不是最终采用结论。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
重点核对当前MMAC是否位于原ready之前、未来LDS读取是否与MMAC重叠、LG0是否仍在下一barrier之前；结合实际VGPR/驻留与访存计数解释收益和回退。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。
正式成绩来自独立七block，perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID621 base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3phase_id621_base_cu0_c0522550/619686_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-0.092%；平均驻留wave/CU 6.331196981605448；HBM读取 154019840字节；wait/wavecycle 0.16081529860934654。
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
## ID621 candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6026；GPU4/epoch0。
- 文件 `captures/h3phase_id621_candidate_cu0_28d6b891/619720_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeCompactPhaseKernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 136, 'sgpr': 64}。
- 正式candidate/base变化-0.092%；平均驻留wave/CU 5.9556680676797065；HBM读取 148784000字节；wait/wavecycle 0.08490040705090804。
- 主循环PC `[{'start_pc': '0x43b9c', 'end_pc': '0x441d4', 'line_start': 286, 'line_end': 518}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 000000043B9C: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 000000043BA4: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[89:90], v[41:42], v[21:24]  // 000000043BAC: FFD40015 04565359`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[89:90], v[43:44], v[17:20]  // 000000043BB4: FFD40011 04465759`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[87:88], v[45:46], v[13:16]  // 000000043BBC: FFD4000D 04365B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[87:88], v[47:48], v[9:12]  // 000000043BC4: FFD40009 04265F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[87:88], v[41:42], v[5:8]  // 000000043BCC: FFD40005 04165357`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[87:88], v[43:44], v[1:4]  // 000000043BD4: FFD40001 04065757`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[85:86], v[33:34], v[25:28]  // 000000043BDC: FFD40019 04664355`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[85:86], v[35:36], v[29:32]  // 000000043BE4: FFD4001D 04764755`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[85:86], v[37:38], v[21:24]  // 000000043BEC: FFD40015 04564B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[85:86], v[39:40], v[17:20]  // 000000043BF4: FFD40011 04464F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[83:84], v[33:34], v[13:16]  // 000000043BFC: FFD4000D 04364353`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[83:84], v[35:36], v[9:12]  // 000000043C04: FFD40009 04264753`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[37:38], v[5:8]  // 000000043C0C: FFD40005 04164B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[39:40], v[1:4]  // 000000043C14: FFD40001 04064F53`
## ID805 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3phase_id805_base_cu0_a613ab01/619761_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-1.952%；平均驻留wave/CU 6.569627332635583；HBM读取 560635712字节；wait/wavecycle 0.1752538951527709。
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
## ID805 candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6026；GPU4/epoch0。
- 文件 `captures/h3phase_id805_candidate_cu0_2d1f6323/619815_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeCompactPhaseKernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 136, 'sgpr': 64}。
- 正式candidate/base变化-1.952%；平均驻留wave/CU 6.743836416726896；HBM读取 641417920字节；wait/wavecycle 0.16561298595182028。
- 主循环PC `[{'start_pc': '0x43b9c', 'end_pc': '0x441d4', 'line_start': 286, 'line_end': 518}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 000000043B9C: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 000000043BA4: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[89:90], v[41:42], v[21:24]  // 000000043BAC: FFD40015 04565359`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[89:90], v[43:44], v[17:20]  // 000000043BB4: FFD40011 04465759`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[87:88], v[45:46], v[13:16]  // 000000043BBC: FFD4000D 04365B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[87:88], v[47:48], v[9:12]  // 000000043BC4: FFD40009 04265F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[87:88], v[41:42], v[5:8]  // 000000043BCC: FFD40005 04165357`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[87:88], v[43:44], v[1:4]  // 000000043BD4: FFD40001 04065757`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[85:86], v[33:34], v[25:28]  // 000000043BDC: FFD40019 04664355`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[85:86], v[35:36], v[29:32]  // 000000043BE4: FFD4001D 04764755`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[85:86], v[37:38], v[21:24]  // 000000043BEC: FFD40015 04564B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[85:86], v[39:40], v[17:20]  // 000000043BF4: FFD40011 04464F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[83:84], v[33:34], v[13:16]  // 000000043BFC: FFD4000D 04364353`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[83:84], v[35:36], v[9:12]  // 000000043C04: FFD40009 04264753`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[37:38], v[5:8]  // 000000043C0C: FFD40005 04164B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[39:40], v[1:4]  // 000000043C14: FFD40001 04064F53`
## ID1009 base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例5002；GPU4/epoch0。
- 文件 `captures/h3phase_id1009_base_cu0_aea8ee11/619866_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-7.072%；平均驻留wave/CU 6.552092993938573；HBM读取 807661120字节；wait/wavecycle 0.19707414775396376。
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
## ID1009 candidate

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6026；GPU4/epoch0。
- 文件 `captures/h3phase_id1009_candidate_cu0_fab1f42c/619933_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeCompactPhaseKernel<1>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 26624, 'scratch_bytes': 0, 'vgpr': 64, 'sgpr': 64}。
- 正式candidate/base变化-7.072%；平均驻留wave/CU 7.084654478919378；HBM读取 805823040字节；wait/wavecycle 0.05399542788037124。
- 主循环PC `[{'start_pc': '0x15b9c', 'end_pc': '0x15ed8', 'line_start': 233, 'line_end': 370}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[35:36], v[13:14], v[1:4]  // 000000015B9C: FFD40001 04061B23`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[35:36], v[15:16], v[5:8]  // 000000015BA8: FFD40005 04161F23`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[9:10], v[1:4]  // 000000015BB0: FFD40001 04061325`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[11:12], v[5:8]  // 000000015BB8: FFD40005 04161725`
  指令PC：`s_waitcnt vmcnt(2)                                         // 000000015BC0: BF8C0F72`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000015BC4: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000015BC8: BF8A0000`
  指令PC：`ds_read_b64 v[35:36], v54                                  // 000000015BCC: D8EC0000 23000036`
  指令PC：`ds_read_m32x16_b16 v[25:28], v61                           // 000000015BD4: D9F60000 1900003D`
  指令PC：`ds_read_b64 v[31:32], v56                                  // 000000015BDC: D8EC0000 1F000038`
  指令PC：`ds_read_m32x16_b16 v[9:12], v62                            // 000000015BE8: D9F60000 0900003E`
  指令PC：`buffer_load_dwordx4 v[33:34], s[8:11], s47 idxen offen slc lds// 000000015C00: E05F3000 2F020021`
  指令PC：`buffer_load_dwordx4 v58, s[16:19], s47 offen lds           // 000000015C10: E05D1000 2F04003A`
  指令PC：`buffer_load_dwordx4 v58, s[16:19], s47 offen lds           // 000000015C20: E05D1000 2F04003A`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[21:22], v[1:4]  // 000000015C28: FFD40001 04062B29`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[23:24], v[5:8]  // 000000015C30: FFD40005 04162F29`
## ID312 base

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例5000；GPU4/epoch0。
- 文件 `captures/h3phase_id312_base_cu0_6f38ca60/619983_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-3.091%；平均驻留wave/CU 7.716696880629563；HBM读取 1091452416字节；wait/wavecycle 0.2083077997983352。
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
## ID312 candidate

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例6026；GPU4/epoch0。
- 文件 `captures/h3phase_id312_candidate_cu0_4714abfb/620074_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeCompactPhaseKernel<2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 2048, 'lds_bytes': 28672, 'scratch_bytes': 0, 'vgpr': 88, 'sgpr': 64}。
- 正式candidate/base变化-3.091%；平均驻留wave/CU 7.651019892340399；HBM读取 1103814336字节；wait/wavecycle 0.07861627614654156。
- 主循环PC `[{'start_pc': '0x29800', 'end_pc': '0x29c58', 'line_start': 251, 'line_end': 423}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[49:50], v[21:22], v[9:12]  // 000000029800: FFD40009 04262B31`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[49:50], v[23:24], v[13:16]  // 000000029808: FFD4000D 04362F31`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[43:44], v[21:22], v[5:8]  // 000000029810: FFD40005 04162B2B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[43:44], v[23:24], v[1:4]  // 000000029818: FFD40001 04062F2B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[45:46], v[17:18], v[9:12]  // 000000029820: FFD40009 0426232D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[45:46], v[19:20], v[13:16]  // 000000029828: FFD4000D 0436272D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[47:48], v[17:18], v[5:8]  // 000000029830: FFD40005 0416232F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[47:48], v[19:20], v[1:4]  // 000000029838: FFD40001 0406272F`
  指令PC：`s_waitcnt vmcnt(2)                                         // 000000029840: BF8C0F72`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000029844: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000029848: BF8A0000`
  指令PC：`ds_read_b64 v[49:50], v77                                  // 00000002984C: D8EC0000 3100004D`
  指令PC：`ds_read_b64 v[45:46], v78                                  // 000000029854: D8EC0000 2D00004E`
  指令PC：`ds_read_m32x16_b16 v[33:36], v84                           // 00000002985C: D9F60000 21000054`
  指令PC：`ds_read_b64 v[43:44], v80                                  // 000000029864: D8EC0000 2B000050`
  指令PC：`ds_read_b64 v[39:40], v79                                  // 00000002986C: D8EC0000 2700004F`
## ID933 base

- MNKG {'M': 64, 'N': 4096, 'K': 4096, 'G': 64}；实例5000；GPU4/epoch0。
- 文件 `captures/h3phase_id933_base_cu0_38514585/620141_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+22.995%；平均驻留wave/CU 7.799244141331932；HBM读取 2192139392字节；wait/wavecycle 0.19490518081390942。
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
## ID933 candidate

- MNKG {'M': 64, 'N': 4096, 'K': 4096, 'G': 64}；实例6026；GPU4/epoch0。
- 文件 `captures/h3phase_id933_candidate_cu0_989899b3/620270_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeCompactPhaseKernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 2048, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 136, 'sgpr': 64}。
- 正式candidate/base变化+22.995%；平均驻留wave/CU 7.696011090426689；HBM读取 2375882368字节；wait/wavecycle 0.07404391312579145。
- 主循环PC `[{'start_pc': '0x43b9c', 'end_pc': '0x441d4', 'line_start': 286, 'line_end': 518}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 000000043B9C: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 000000043BA4: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[89:90], v[41:42], v[21:24]  // 000000043BAC: FFD40015 04565359`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[89:90], v[43:44], v[17:20]  // 000000043BB4: FFD40011 04465759`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[87:88], v[45:46], v[13:16]  // 000000043BBC: FFD4000D 04365B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[87:88], v[47:48], v[9:12]  // 000000043BC4: FFD40009 04265F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[87:88], v[41:42], v[5:8]  // 000000043BCC: FFD40005 04165357`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[87:88], v[43:44], v[1:4]  // 000000043BD4: FFD40001 04065757`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[85:86], v[33:34], v[25:28]  // 000000043BDC: FFD40019 04664355`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[85:86], v[35:36], v[29:32]  // 000000043BE4: FFD4001D 04764755`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[85:86], v[37:38], v[21:24]  // 000000043BEC: FFD40015 04564B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[85:86], v[39:40], v[17:20]  // 000000043BF4: FFD40011 04464F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[83:84], v[33:34], v[13:16]  // 000000043BFC: FFD4000D 04364353`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[83:84], v[35:36], v[9:12]  // 000000043C04: FFD40009 04264753`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[37:38], v[5:8]  // 000000043C0C: FFD40005 04164B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[39:40], v[1:4]  // 000000043C14: FFD40001 04064F53`
