# K64双槽与当前最佳CK对照

全部46个预登记发现配对结束后采集；固定621/805/1009/312并补齐实际存在的收益/平坦/回退类别。候选选择和这些诊断样本均不是最终采用结论。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
重点核对当前MMAC是否位于原ready之前、未来LDS读取是否与MMAC重叠、LG0是否仍在下一barrier之前；结合实际VGPR/驻留与访存计数解释收益和回退。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。
本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID621 base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3k64_id621_base_cu0_74632680/694178_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-2.362%；平均驻留wave/CU 6.360953105884482；HBM读取 156670016字节；wait/wavecycle 0.15423351204988714。
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

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6028；GPU4/epoch0。
- 文件 `captures/h3k64_id621_candidate_cu0_3f48826d/694221_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeK64Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 220, 'sgpr': 80}。
- 正式candidate/base变化-2.362%；平均驻留wave/CU 3.168729033256976；HBM读取 146739008字节；wait/wavecycle 0.24713179832280363。
- 主循环PC `[{'start_pc': '0x4e1e8', 'end_pc': '0x4eddc', 'line_start': 380, 'line_end': 816}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[163:164], v[93:94], v[25:28]  // 00000004E1E8: FFD40019 0466BBA3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[163:164], v[95:96], v[29:32]  // 00000004E1F0: FFD4001D 0476BFA3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[163:164], v[89:90], v[21:24]  // 00000004E1F8: FFD40015 0456B3A3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[163:164], v[91:92], v[17:20]  // 00000004E200: FFD40011 0446B7A3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[161:162], v[93:94], v[13:16]  // 00000004E208: FFD4000D 0436BBA1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[161:162], v[95:96], v[9:12]  // 00000004E210: FFD40009 0426BFA1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[161:162], v[89:90], v[5:8]  // 00000004E218: FFD40005 0416B3A1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[161:162], v[91:92], v[1:4]  // 00000004E220: FFD40001 0406B7A1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[159:160], v[77:78], v[25:28]  // 00000004E228: FFD40019 04669B9F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[159:160], v[79:80], v[29:32]  // 00000004E230: FFD4001D 04769F9F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[159:160], v[81:82], v[21:24]  // 00000004E238: FFD40015 0456A39F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[159:160], v[83:84], v[17:20]  // 00000004E240: FFD40011 0446A79F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[157:158], v[77:78], v[13:16]  // 00000004E248: FFD4000D 04369B9D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[157:158], v[79:80], v[9:12]  // 00000004E250: FFD40009 04269F9D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[157:158], v[81:82], v[5:8]  // 00000004E258: FFD40005 0416A39D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[157:158], v[83:84], v[1:4]  // 00000004E260: FFD40001 0406A79D`
## ID805 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3k64_id805_base_cu0_9108973e/694264_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-0.684%；平均驻留wave/CU 6.6755641258155585；HBM读取 570282432字节；wait/wavecycle 0.1726206830872364。
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

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6028；GPU4/epoch0。
- 文件 `captures/h3k64_id805_candidate_cu0_01935120/694317_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeK64Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 220, 'sgpr': 80}。
- 正式candidate/base变化-0.684%；平均驻留wave/CU 3.6261356088277723；HBM读取 563003968字节；wait/wavecycle 0.257382880641881。
- 主循环PC `[{'start_pc': '0x4e1e8', 'end_pc': '0x4eddc', 'line_start': 380, 'line_end': 816}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[163:164], v[93:94], v[25:28]  // 00000004E1E8: FFD40019 0466BBA3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[163:164], v[95:96], v[29:32]  // 00000004E1F0: FFD4001D 0476BFA3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[163:164], v[89:90], v[21:24]  // 00000004E1F8: FFD40015 0456B3A3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[163:164], v[91:92], v[17:20]  // 00000004E200: FFD40011 0446B7A3`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[161:162], v[93:94], v[13:16]  // 00000004E208: FFD4000D 0436BBA1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[161:162], v[95:96], v[9:12]  // 00000004E210: FFD40009 0426BFA1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[161:162], v[89:90], v[5:8]  // 00000004E218: FFD40005 0416B3A1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[161:162], v[91:92], v[1:4]  // 00000004E220: FFD40001 0406B7A1`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[159:160], v[77:78], v[25:28]  // 00000004E228: FFD40019 04669B9F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[159:160], v[79:80], v[29:32]  // 00000004E230: FFD4001D 04769F9F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[159:160], v[81:82], v[21:24]  // 00000004E238: FFD40015 0456A39F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[159:160], v[83:84], v[17:20]  // 00000004E240: FFD40011 0446A79F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[157:158], v[77:78], v[13:16]  // 00000004E248: FFD4000D 04369B9D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[157:158], v[79:80], v[9:12]  // 00000004E250: FFD40009 04269F9D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[157:158], v[81:82], v[5:8]  // 00000004E258: FFD40005 0416A39D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[157:158], v[83:84], v[1:4]  // 00000004E260: FFD40001 0406A79D`
## ID1009 base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例5002；GPU4/epoch0。
- 文件 `captures/h3k64_id1009_base_cu0_50987bd7/694371_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-12.092%；平均驻留wave/CU 6.5496386286998876；HBM读取 807717376字节；wait/wavecycle 0.19983484919516453。
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

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6028；GPU4/epoch0。
- 文件 `captures/h3k64_id1009_candidate_cu0_a40e1c2d/694438_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeK64Kernel<1>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 36864, 'scratch_bytes': 0, 'vgpr': 104, 'sgpr': 80}。
- 正式candidate/base变化-12.092%；平均驻留wave/CU 3.6950718462287697；HBM读取 805867136字节；wait/wavecycle 0.07469169423127338。
- 主循环PC `[{'start_pc': '0x184c0', 'end_pc': '0x18a40', 'line_start': 272, 'line_end': 493}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[71:72], v[37:38], v[1:4]  // 0000000184C0: FFD40001 04064B47`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[71:72], v[39:40], v[5:8]  // 0000000184CC: FFD40005 04164F47`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[73:74], v[33:34], v[1:4]  // 0000000184D4: FFD40001 04064349`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[73:74], v[35:36], v[5:8]  // 0000000184DC: FFD40005 04164749`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[69:70], v[29:30], v[1:4]  // 0000000184E4: FFD40001 04063B45`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[69:70], v[31:32], v[5:8]  // 0000000184EC: FFD40005 04163F45`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[67:68], v[25:26], v[1:4]  // 0000000184F4: FFD40001 04063343`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[67:68], v[27:28], v[5:8]  // 0000000184FC: FFD40005 04163743`
  指令PC：`s_waitcnt vmcnt(0)                                         // 000000018504: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000018508: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000001850C: BF8A0000`
  指令PC：`ds_read_b64 v[69:70], v94                                  // 000000018510: D8EC0000 4500005E`
  指令PC：`ds_read_m32x16_b16 v[41:44], v95                           // 000000018518: D9F60000 2900005F`
  指令PC：`ds_read_b64 v[67:68], v98                                  // 000000018520: D8EC0000 43000062`
  指令PC：`ds_read_m32x16_b16 v[33:36], v96                           // 000000018528: D9F60000 21000060`
  指令PC：`ds_read_b64 v[53:54], v100                                 // 000000018530: D8EC0000 35000064`
## ID312 base

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例5000；GPU4/epoch0。
- 文件 `captures/h3k64_id312_base_cu0_6446987d/694485_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+8.322%；平均驻留wave/CU 7.712301349387837；HBM读取 1091561216字节；wait/wavecycle 0.20524421456984374。
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

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例6028；GPU4/epoch0。
- 文件 `captures/h3k64_id312_candidate_cu0_2c2d40f7/694572_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::DecodeNnCooperativeK64Kernel<2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 2048, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 144, 'sgpr': 80}。
- 正式candidate/base变化+8.322%；平均驻留wave/CU 3.8674416303286065；HBM读取 1096742272字节；wait/wavecycle 0.14821536425991833。
- 主循环PC `[{'start_pc': '0x2fd74', 'end_pc': '0x30530', 'line_start': 309, 'line_end': 601}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[97:98], v[45:46], v[9:12]  // 00000002FD74: FFD40009 04265B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[97:98], v[47:48], v[13:16]  // 00000002FD7C: FFD4000D 04365F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[45:46], v[5:8]  // 00000002FD84: FFD40005 04165B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[47:48], v[1:4]  // 00000002FD8C: FFD40001 04065F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[41:42], v[9:12]  // 00000002FD94: FFD40009 0426535B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[43:44], v[13:16]  // 00000002FD9C: FFD4000D 0436575B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[93:94], v[41:42], v[5:8]  // 00000002FDA4: FFD40005 0416535D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[93:94], v[43:44], v[1:4]  // 00000002FDAC: FFD40001 0406575D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[87:88], v[37:38], v[9:12]  // 00000002FDB4: FFD40009 04264B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[87:88], v[39:40], v[13:16]  // 00000002FDBC: FFD4000D 04364F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[89:90], v[37:38], v[5:8]  // 00000002FDC4: FFD40005 04164B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[89:90], v[39:40], v[1:4]  // 00000002FDCC: FFD40001 04064F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[85:86], v[33:34], v[9:12]  // 00000002FDD4: FFD40009 04264355`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[85:86], v[35:36], v[13:16]  // 00000002FDDC: FFD4000D 04364755`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[33:34], v[5:8]  // 00000002FDE4: FFD40005 04164353`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[35:36], v[1:4]  // 00000002FDEC: FFD40001 04064753`
