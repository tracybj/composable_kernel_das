# B复用的双槽/三槽与当前CK、原消费者分工对照

全部59个预登记发现配对结束后采集；固定621/805/1009/312并补齐实际存在的收益/平坦/回退类别，另固定ID805在所选深度的原分工/新B复用直接控制。主对照base是当前最佳CK；partition_control的base是原N256实验核6030或6031，candidate是同深度B复用6032或6033，不能混称为当前最佳CK。这些诊断样本均不是最终采用结论。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
重点比较ID805同深度分工控制：全局A/B生产者与VM请求计数不变；M64每wave的B读取减半、A读取加倍、总LDS字节9→6KiB/K32，MMAC计数不变。结合实际VGPR/LDS读取、驻留及等待，检查未来操作数与当前MMAC衔接。D2/D3依旧分别vmcnt(0)和vmcnt(2/3)。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。
本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID621 id621_d2 base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id621_base_cu0_37c2ff4a/845123_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-9.261%；平均驻留wave/CU 6.377991705590833；HBM读取 156800896字节；wait/wavecycle 0.13954352532324077。
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
## ID621 id621_d2 candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id621_candidate_cu0_c5633520/845157_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 128, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化-9.261%；平均驻留wave/CU 6.115865766214198；HBM读取 147042304字节；wait/wavecycle 0.12231193730616859。
- 主循环PC `[{'start_pc': '0x64d44', 'end_pc': '0x6537c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000064D44: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 000000064D4C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000064D54: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 000000064D5C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000064D64: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 000000064D6C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000064D74: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 000000064D7C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000064D84: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 000000064D8C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000064D94: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 000000064D9C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 000000064DA4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 000000064DAC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 000000064DB4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 000000064DBC: FFD40001 04066F5B`
## ID805 id805_d2 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id805_base_cu0_acd841a4/845193_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-4.613%；平均驻留wave/CU 6.68156630336113；HBM读取 568501248字节；wait/wavecycle 0.16705337535616976。
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
## ID805 id805_d2 candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id805_candidate_cu0_e7b7d21f/845247_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化-4.613%；平均驻留wave/CU 6.687593629655858；HBM读取 563055040字节；wait/wavecycle 0.12025376702516674。
- 主循环PC `[{'start_pc': '0x64d44', 'end_pc': '0x6537c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000064D44: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 000000064D4C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000064D54: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 000000064D5C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000064D64: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 000000064D6C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000064D74: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 000000064D7C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000064D84: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 000000064D8C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000064D94: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 000000064D9C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 000000064DA4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 000000064DAC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 000000064DB4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 000000064DBC: FFD40001 04066F5B`
## ID1009 id1009_d2 base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例5002；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id1009_base_cu0_31fb0587/845294_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-8.267%；平均驻留wave/CU 6.528451600688694；HBM读取 807674560字节；wait/wavecycle 0.19731964223939835。
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
## ID1009 id1009_d2 candidate

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id1009_candidate_cu0_636d15a8/845368_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<1, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 34816, 'scratch_bytes': 0, 'vgpr': 64, 'sgpr': 48}。
- 正式candidate/base变化-8.267%；平均驻留wave/CU 6.7633259068195075；HBM读取 805633728字节；wait/wavecycle 0.061649106790868276。
- 主循环PC `[{'start_pc': '0x1f348', 'end_pc': '0x1f684', 'line_start': 220, 'line_end': 357}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[39:40], v[21:22], v[1:4]  // 00000001F348: FFD40001 04062B27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[39:40], v[23:24], v[5:8]  // 00000001F354: FFD40005 04162F27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[17:18], v[1:4]  // 00000001F35C: FFD40001 04062329`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[19:20], v[5:8]  // 00000001F364: FFD40005 04162729`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000001F36C: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000001F370: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000001F374: BF8A0000`
  指令PC：`ds_read_b64 v[41:42], v56                                  // 00000001F378: D8EC0000 29000038`
  指令PC：`ds_read_m32x16_b16 v[25:28], v57                           // 00000001F380: D9F60000 19000039`
  指令PC：`ds_read_b64 v[31:32], v59                                  // 00000001F388: D8EC0000 1F00003B`
  指令PC：`ds_read_m32x16_b16 v[17:20], v58                           // 00000001F394: D9F60000 1100003A`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s42 idxen offen slc lds// 00000001F3AC: E05F3000 2A040021`
  指令PC：`buffer_load_dwordx4 v55, s[12:15], s42 offen lds           // 00000001F3BC: E05D1000 2A030037`
  指令PC：`buffer_load_dwordx4 v55, s[12:15], s42 offen lds           // 00000001F3CC: E05D1000 2A030037`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 00000001F3D4: FFD40001 04061B25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[15:16], v[5:8]  // 00000001F3DC: FFD40005 04161F25`
## ID312 id312_d2 base

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例5000；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id312_base_cu0_cebd4680/845419_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-6.301%；平均驻留wave/CU 7.7971758470070505；HBM读取 1092017024字节；wait/wavecycle 0.17392837295349442。
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
## ID312 id312_d2 candidate

- MNKG {'M': 32, 'N': 4096, 'K': 2048, 'G': 64}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id312_candidate_cu0_30bc2ed6/845511_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<2, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [524288, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 1024, 'lds_bytes': 36864, 'scratch_bytes': 0, 'vgpr': 88, 'sgpr': 64}。
- 正式candidate/base变化-6.301%；平均驻留wave/CU 7.7466405259083695；HBM读取 1086809536字节；wait/wavecycle 0.07871419806079212。
- 主循环PC `[{'start_pc': '0x3ddb0', 'end_pc': '0x3e208', 'line_start': 240, 'line_end': 412}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[57:58], v[29:30], v[9:12]  // 00000003DDB0: FFD40009 04263B39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[57:58], v[31:32], v[13:16]  // 00000003DDB8: FFD4000D 04363F39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[55:56], v[29:30], v[5:8]  // 00000003DDC0: FFD40005 04163B37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[55:56], v[31:32], v[1:4]  // 00000003DDC8: FFD40001 04063F37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[53:54], v[25:26], v[9:12]  // 00000003DDD0: FFD40009 04263335`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[53:54], v[27:28], v[13:16]  // 00000003DDD8: FFD4000D 04363735`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[51:52], v[25:26], v[5:8]  // 00000003DDE0: FFD40005 04163333`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[51:52], v[27:28], v[1:4]  // 00000003DDE8: FFD40001 04063733`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000003DDF0: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000003DDF4: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000003DDF8: BF8A0000`
  指令PC：`ds_read_b64 v[59:60], v78                                  // 00000003DDFC: D8EC0000 3B00004E`
  指令PC：`ds_read_b64 v[55:56], v79                                  // 00000003DE04: D8EC0000 3700004F`
  指令PC：`ds_read_m32x16_b16 v[33:36], v80                           // 00000003DE0C: D9F60000 21000050`
  指令PC：`ds_read_b64 v[51:52], v83                                  // 00000003DE14: D8EC0000 33000053`
  指令PC：`ds_read_b64 v[39:40], v82                                  // 00000003DE1C: D8EC0000 27000052`
## ID328 id328_d2 base

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 16}；实例5003；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id328_base_cu0_d6cf23d2/845571_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+0.325%；平均驻留wave/CU 6.769830262420275；HBM读取 278239360字节；wait/wavecycle 0.15977051769625708。
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
## ID328 id328_d2 candidate

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 16}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id328_candidate_cu0_f147421d/845618_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化+0.325%；平均驻留wave/CU 6.571037291533174；HBM读取 274736768字节；wait/wavecycle 0.1296365772319652。
- 主循环PC `[{'start_pc': '0x64d44', 'end_pc': '0x6537c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000064D44: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 000000064D4C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000064D54: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 000000064D5C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000064D64: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 000000064D6C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000064D74: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 000000064D7C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000064D84: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 000000064D8C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000064D94: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 000000064D9C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 000000064DA4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 000000064DAC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 000000064DB4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 000000064DBC: FFD40001 04066F5B`
## ID335 id335_d2 base

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例5003；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id335_base_cu0_cf1ccce7/845652_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+6.326%；平均驻留wave/CU 5.987052464666002；HBM读取 139143744字节；wait/wavecycle 0.1435305236520928。
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
## ID335 id335_d2 candidate

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id335_candidate_cu0_65687696/845692_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 128, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化+6.326%；平均驻留wave/CU 6.04497924969613；HBM读取 137651904字节；wait/wavecycle 0.14513904660930096。
- 主循环PC `[{'start_pc': '0x64d44', 'end_pc': '0x6537c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000064D44: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 000000064D4C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000064D54: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 000000064D5C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000064D64: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 000000064D6C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000064D74: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 000000064D7C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000064D84: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 000000064D8C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000064D94: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 000000064D9C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 000000064DA4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 000000064DAC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 000000064DB4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 000000064DBC: FFD40001 04066F5B`
## ID805 partition_id805_d2 base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6030；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id805_base_cu0_32623ed2/845726_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 168, 'sgpr': 64}。
- 正式candidate/base变化-10.060%；平均驻留wave/CU 6.697296331747868；HBM读取 570368320字节；wait/wavecycle 0.15271745282439741。
- 主循环PC `[{'start_pc': '0x60788', 'end_pc': '0x60e20', 'line_start': 282, 'line_end': 526}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[137:138], v[93:94], v[25:28]  // 000000060788: FFD40019 0466BB89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[137:138], v[95:96], v[29:32]  // 000000060790: FFD4001D 0476BF89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[137:138], v[89:90], v[21:24]  // 000000060798: FFD40015 0456B389`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[137:138], v[91:92], v[17:20]  // 0000000607A0: FFD40011 0446B789`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[137:138], v[85:86], v[13:16]  // 0000000607A8: FFD4000D 0436AB89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[137:138], v[87:88], v[9:12]  // 0000000607B0: FFD40009 0426AF89`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[137:138], v[81:82], v[5:8]  // 0000000607B8: FFD40005 0416A389`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[137:138], v[83:84], v[1:4]  // 0000000607C0: FFD40001 0406A789`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[135:136], v[77:78], v[25:28]  // 0000000607C8: FFD40019 04669B87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[135:136], v[79:80], v[29:32]  // 0000000607D0: FFD4001D 04769F87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[135:136], v[73:74], v[21:24]  // 0000000607D8: FFD40015 04569387`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[135:136], v[75:76], v[17:20]  // 0000000607E0: FFD40011 04469787`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[135:136], v[69:70], v[13:16]  // 0000000607E8: FFD4000D 04368B87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[135:136], v[71:72], v[9:12]  // 0000000607F0: FFD40009 04268F87`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[135:136], v[65:66], v[5:8]  // 0000000607F8: FFD40005 04168387`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[135:136], v[67:68], v[1:4]  // 000000060800: FFD40001 04068787`
## ID805 partition_id805_d2 candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6032；GPU4/epoch0。
- 文件 `captures/h3b_reuse_id805_candidate_cu0_57da62e1/845774_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化-10.060%；平均驻留wave/CU 6.687919140438521；HBM读取 562248896字节；wait/wavecycle 0.11959524522837055。
- 主循环PC `[{'start_pc': '0x64d44', 'end_pc': '0x6537c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000064D44: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 000000064D4C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000064D54: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 000000064D5C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000064D64: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 000000064D6C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000064D74: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 000000064D7C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000064D84: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 000000064D8C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000064D94: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 000000064D9C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 000000064DA4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 000000064DAC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 000000064DB4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 000000064DBC: FFD40001 04066F5B`
