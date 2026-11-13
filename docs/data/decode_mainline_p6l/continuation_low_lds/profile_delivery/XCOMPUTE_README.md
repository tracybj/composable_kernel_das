# 低LDS N128双槽：当前CK、纯布局与自动属性规则对照

全部59个预登记发现配对结束后采集；在两个相同23代表集合上选择诊断policy；固定621/805/1017/686并补齐实际存在的收益/平坦/回退类别，另固定ID805及ID1009的N256D2 B复用6032/新N128D2 6034几何控制。主对照base是当前最佳CK；geometry_control的base是独立N256D2实验核，不能混称为当前最佳CK。N宽、wave分工和LDS足迹同时改变，不称为纯N因果。这些诊断样本均不是最终采用结论。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
重点比较LDS18/20/24KiB能否带来更多实测驻留wave、更多有效CTA及请求重叠。相对N256实验核，N128使CTA数增加，同时A生产者和部分LDS消费者访问增加；核对HBM是否仍接近一份输入。主循环K32/D2/三操作数bank不变，vmcnt(0)及LG0/barrier必须保留。M<=16为4wave，其余8wave。平均驻留数本身不是理论上限证明；长wait与不同wave累计Latency不能直接相加当kernel墙钟。
本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID621 id621_auto base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3low_lds_id621_base_cu0_941ae637/923758_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-2.832%；平均驻留wave/CU 6.376525350524136；HBM读取 156221632字节；wait/wavecycle 0.15359333028217792。
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
## ID621 id621_auto candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id621_candidate_cu0_fe44cc01/923793_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnLowLdsN128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 84, 'sgpr': 48}。
- 正式candidate/base变化-2.832%；平均驻留wave/CU 12.567967631661118；HBM读取 154639040字节；wait/wavecycle 0.0852374018466558。
- 主循环PC `[{'start_pc': '0x6b938', 'end_pc': '0x6bd30', 'line_start': 222, 'line_end': 376}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[57:58], v[29:30], v[9:12]  // 00000006B938: FFD40009 04263B39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[57:58], v[31:32], v[13:16]  // 00000006B940: FFD4000D 04363F39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[55:56], v[29:30], v[5:8]  // 00000006B948: FFD40005 04163B37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[55:56], v[31:32], v[1:4]  // 00000006B950: FFD40001 04063F37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[53:54], v[25:26], v[9:12]  // 00000006B958: FFD40009 04263335`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[53:54], v[27:28], v[13:16]  // 00000006B960: FFD4000D 04363735`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[51:52], v[25:26], v[5:8]  // 00000006B968: FFD40005 04163333`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[51:52], v[27:28], v[1:4]  // 00000006B970: FFD40001 04063733`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000006B978: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000006B97C: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000006B980: BF8A0000`
  指令PC：`ds_read_b64 v[55:56], v77                                  // 00000006B984: D8EC0000 3700004D`
  指令PC：`ds_read_b64 v[53:54], v78                                  // 00000006B98C: D8EC0000 3500004E`
  指令PC：`ds_read_m32x16_b16 v[29:32], v80                           // 00000006B994: D9F60000 1D000050`
  指令PC：`ds_read_b64 v[51:52], v82                                  // 00000006B99C: D8EC0000 33000052`
  指令PC：`ds_read_b64 v[37:38], v81                                  // 00000006B9A4: D8EC0000 25000051`
## ID805 id805_auto base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3low_lds_id805_base_cu0_bbc7ab93/923833_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化-3.753%；平均驻留wave/CU 6.567381956568648；HBM读取 560106624字节；wait/wavecycle 0.17181295567210192。
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
## ID805 id805_auto candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6032；GPU4/epoch0。
- 文件 `captures/h3low_lds_id805_candidate_cu0_6997ec2e/923883_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化-3.753%；平均驻留wave/CU 6.684319204698538；HBM读取 561745280字节；wait/wavecycle 0.12046380535152426。
- 主循环PC `[{'start_pc': '0x67744', 'end_pc': '0x67d7c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000067744: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 00000006774C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000067754: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 00000006775C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000067764: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 00000006776C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000067774: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 00000006777C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000067784: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 00000006778C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000067794: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 00000006779C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 0000000677A4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 0000000677AC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 0000000677B4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 0000000677BC: FFD40001 04066F5B`
## ID1017 id1017_auto base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 8}；实例5006；GPU4/epoch0。
- 文件 `captures/h3low_lds_id1017_base_cu0_b71e4378/923934_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 8>, ck_tile::GemmPipelineV7CacheNoSlc<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 128, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+23.541%；平均驻留wave/CU 6.409130662885232；HBM读取 403828608字节；wait/wavecycle 0.19768359127495513。
- 主循环PC `[{'start_pc': '0x2d20', 'end_pc': '0x3ff0', 'static_mmac': 384, 'line_start': 659, 'line_end': 1360}, {'start_pc': '0x2ce0', 'end_pc': '0x3fec', 'static_mmac': 384, 'line_start': 651, 'line_end': 1359}, {'start_pc': '0x5528', 'end_pc': '0x683c', 'static_mmac': 384, 'line_start': 2256, 'line_end': 2966}, {'start_pc': '0x4150', 'end_pc': '0x4fd4', 'static_mmac': 288, 'line_start': 1448, 'line_end': 1999}]`。

  指令PC：`buffer_load_dwordx4 v[177:178], s[4:7], s13 idxen offen lds// 000000002D44: E05D3000 0D0100B1`
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
## ID1017 id1017_auto candidate

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 8}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id1017_candidate_cu0_292e7277/923987_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsN128Kernel<1>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 18432, 'scratch_bytes': 0, 'vgpr': 60, 'sgpr': 64}。
- 正式candidate/base变化+23.541%；平均驻留wave/CU 9.26697173521232；HBM读取 403859136字节；wait/wavecycle 0.02165529306813908。
- 主循环PC `[{'start_pc': '0x22748', 'end_pc': '0x22a84', 'line_start': 220, 'line_end': 357}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[39:40], v[21:22], v[1:4]  // 000000022748: FFD40001 04062B27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[39:40], v[23:24], v[5:8]  // 000000022754: FFD40005 04162F27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[17:18], v[1:4]  // 00000002275C: FFD40001 04062329`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[19:20], v[5:8]  // 000000022764: FFD40005 04162729`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000002276C: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000022770: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000022774: BF8A0000`
  指令PC：`ds_read_b64 v[41:42], v55                                  // 000000022778: D8EC0000 29000037`
  指令PC：`ds_read_m32x16_b16 v[25:28], v56                           // 000000022780: D9F60000 19000038`
  指令PC：`ds_read_b64 v[31:32], v58                                  // 000000022788: D8EC0000 1F00003A`
  指令PC：`ds_read_m32x16_b16 v[17:20], v57                           // 000000022794: D9F60000 11000039`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s43 idxen offen slc lds// 0000000227AC: E05F3000 2B040021`
  指令PC：`buffer_load_dwordx4 v54, s[12:15], s43 offen lds           // 0000000227BC: E05D1000 2B030036`
  指令PC：`buffer_load_dwordx4 v54, s[12:15], s43 offen lds           // 0000000227CC: E05D1000 2B030036`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 0000000227D4: FFD40001 04061B25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[15:16], v[5:8]  // 0000000227DC: FFD40005 04161F25`
## ID686 id686_auto base

- MNKG {'M': 32, 'N': 1536, 'K': 2048, 'G': 4}；实例5008；GPU4/epoch0。
- 文件 `captures/h3low_lds_id686_base_cu1_5955685a/924096_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index1。
- kernel `void ck_tile::kentry<256, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallMK32<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMK32Geometry<128> >, ck_tile::V7SmallMK32Geometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [12288, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 48, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 116, 'sgpr': 64}。
- 正式candidate/base变化+47.916%；平均驻留wave/CU 1.7557411731519643；HBM读取 26329920字节；wait/wavecycle 0.07354162075797469。
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
## ID686 id686_auto candidate

- MNKG {'M': 32, 'N': 1536, 'K': 2048, 'G': 4}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id686_candidate_cu1_f2ad1590/924124_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index1。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnLowLdsN128Kernel<2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [24576, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 48, 'lds_bytes': 20480, 'scratch_bytes': 0, 'vgpr': 60, 'sgpr': 48}。
- 正式candidate/base变化+47.916%；平均驻留wave/CU 3.749945325650256；HBM读取 26208768字节；wait/wavecycle 0.06394071464553647。
- 主循环PC `[{'start_pc': '0x429d4', 'end_pc': '0x42cb0', 'line_start': 203, 'line_end': 322}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[21:22], v[1:4]  // 0000000429D4: FFD40001 04062B29`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[23:24], v[5:8]  // 0000000429E0: FFD40005 04162F29`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[39:40], v[17:18], v[1:4]  // 0000000429E8: FFD40001 04062327`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[39:40], v[19:20], v[5:8]  // 0000000429F0: FFD40005 04162727`
  指令PC：`s_waitcnt vmcnt(0)                                         // 0000000429F8: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 0000000429FC: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000042A00: BF8A0000`
  指令PC：`ds_read_b64 v[39:40], v55                                  // 000000042A04: D8EC0000 27000037`
  指令PC：`ds_read_m32x16_b16 v[21:24], v56                           // 000000042A0C: D9F60000 15000038`
  指令PC：`ds_read_b64 v[29:30], v58                                  // 000000042A14: D8EC0000 1D00003A`
  指令PC：`ds_read_m32x16_b16 v[17:20], v57                           // 000000042A20: D9F60000 11000039`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s34 idxen offen slc lds// 000000042A38: E05F3000 22040021`
  指令PC：`buffer_load_dwordx4 v54, s[12:15], s34 offen lds           // 000000042A48: E05D1000 22030036`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 000000042A50: FFD40001 04061B25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[15:16], v[5:8]  // 000000042A58: FFD40005 04161F25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[35:36], v[9:10], v[1:4]  // 000000042A60: FFD40001 04061323`
## ID335 id335_auto base

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例5003；GPU4/epoch0。
- 文件 `captures/h3low_lds_id335_base_cu0_3c132224/924160_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+0.520%；平均驻留wave/CU 5.939538217801211；HBM读取 139992000字节；wait/wavecycle 0.1464470790454842。
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
## ID335 id335_auto candidate

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id335_candidate_cu0_ea6e51cf/924197_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnLowLdsN128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 84, 'sgpr': 48}。
- 正式candidate/base变化+0.520%；平均驻留wave/CU 12.085214957331893；HBM读取 137916800字节；wait/wavecycle 0.11611114558679048。
- 主循环PC `[{'start_pc': '0x6b938', 'end_pc': '0x6bd30', 'line_start': 222, 'line_end': 376}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[57:58], v[29:30], v[9:12]  // 00000006B938: FFD40009 04263B39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[57:58], v[31:32], v[13:16]  // 00000006B940: FFD4000D 04363F39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[55:56], v[29:30], v[5:8]  // 00000006B948: FFD40005 04163B37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[55:56], v[31:32], v[1:4]  // 00000006B950: FFD40001 04063F37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[53:54], v[25:26], v[9:12]  // 00000006B958: FFD40009 04263335`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[53:54], v[27:28], v[13:16]  // 00000006B960: FFD4000D 04363735`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[51:52], v[25:26], v[5:8]  // 00000006B968: FFD40005 04163333`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[51:52], v[27:28], v[1:4]  // 00000006B970: FFD40001 04063733`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000006B978: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000006B97C: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000006B980: BF8A0000`
  指令PC：`ds_read_b64 v[55:56], v77                                  // 00000006B984: D8EC0000 3700004D`
  指令PC：`ds_read_b64 v[53:54], v78                                  // 00000006B98C: D8EC0000 3500004E`
  指令PC：`ds_read_m32x16_b16 v[29:32], v80                           // 00000006B994: D9F60000 1D000050`
  指令PC：`ds_read_b64 v[51:52], v82                                  // 00000006B99C: D8EC0000 33000052`
  指令PC：`ds_read_b64 v[37:38], v81                                  // 00000006B9A4: D8EC0000 25000051`
## ID805 geometry_id805_pure base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6032；GPU4/epoch0。
- 文件 `captures/h3low_lds_id805_base_cu0_aba85e14/924231_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<4, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 40960, 'scratch_bytes': 0, 'vgpr': 128, 'sgpr': 64}。
- 正式candidate/base变化+26.036%；平均驻留wave/CU 6.6953692800358064；HBM读取 562076544字节；wait/wavecycle 0.1180751971620177。
- 主循环PC `[{'start_pc': '0x67744', 'end_pc': '0x67d7c', 'line_start': 271, 'line_end': 503}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[97:98], v[61:62], v[25:28]  // 000000067744: FFD40019 04667B61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[97:98], v[63:64], v[29:32]  // 00000006774C: FFD4001D 04767F61`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[97:98], v[57:58], v[21:24]  // 000000067754: FFD40015 04567361`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[97:98], v[59:60], v[17:20]  // 00000006775C: FFD40011 04467761`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[95:96], v[61:62], v[13:16]  // 000000067764: FFD4000D 04367B5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[95:96], v[63:64], v[9:12]  // 00000006776C: FFD40009 04267F5F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[95:96], v[57:58], v[5:8]  // 000000067774: FFD40005 0416735F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[95:96], v[59:60], v[1:4]  // 00000006777C: FFD40001 0406775F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[93:94], v[49:50], v[25:28]  // 000000067784: FFD40019 0466635D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[93:94], v[51:52], v[29:32]  // 00000006778C: FFD4001D 0476675D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[93:94], v[53:54], v[21:24]  // 000000067794: FFD40015 04566B5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[93:94], v[55:56], v[17:20]  // 00000006779C: FFD40011 04466F5D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[91:92], v[49:50], v[13:16]  // 0000000677A4: FFD4000D 0436635B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[91:92], v[51:52], v[9:12]  // 0000000677AC: FFD40009 0426675B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[91:92], v[53:54], v[5:8]  // 0000000677B4: FFD40005 04166B5B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[91:92], v[55:56], v[1:4]  // 0000000677BC: FFD40001 04066F5B`
## ID805 geometry_id805_pure candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id805_candidate_cu0_70fc4e47/924278_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnLowLdsN128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [262144, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 512, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 84, 'sgpr': 48}。
- 正式candidate/base变化+26.036%；平均驻留wave/CU 13.563102089139873；HBM读取 635483520字节；wait/wavecycle 0.0969515698315999。
- 主循环PC `[{'start_pc': '0x6b938', 'end_pc': '0x6bd30', 'line_start': 222, 'line_end': 376}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[57:58], v[29:30], v[9:12]  // 00000006B938: FFD40009 04263B39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[57:58], v[31:32], v[13:16]  // 00000006B940: FFD4000D 04363F39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[55:56], v[29:30], v[5:8]  // 00000006B948: FFD40005 04163B37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[55:56], v[31:32], v[1:4]  // 00000006B950: FFD40001 04063F37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[53:54], v[25:26], v[9:12]  // 00000006B958: FFD40009 04263335`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[53:54], v[27:28], v[13:16]  // 00000006B960: FFD4000D 04363735`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[51:52], v[25:26], v[5:8]  // 00000006B968: FFD40005 04163333`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[51:52], v[27:28], v[1:4]  // 00000006B970: FFD40001 04063733`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000006B978: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 00000006B97C: BF8CC07F`
  指令PC：`s_barrier                                                  // 00000006B980: BF8A0000`
  指令PC：`ds_read_b64 v[55:56], v77                                  // 00000006B984: D8EC0000 3700004D`
  指令PC：`ds_read_b64 v[53:54], v78                                  // 00000006B98C: D8EC0000 3500004E`
  指令PC：`ds_read_m32x16_b16 v[29:32], v80                           // 00000006B994: D9F60000 1D000050`
  指令PC：`ds_read_b64 v[51:52], v82                                  // 00000006B99C: D8EC0000 33000052`
  指令PC：`ds_read_b64 v[37:38], v81                                  // 00000006B9A4: D8EC0000 25000051`
## ID1009 geometry_id1009_pure base

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6032；GPU4/epoch0。
- 文件 `captures/h3low_lds_id1009_base_cu0_8c9c9f22/924334_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnBReuseN256Kernel<1, 2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 34816, 'scratch_bytes': 0, 'vgpr': 64, 'sgpr': 48}。
- 正式candidate/base变化+10.407%；平均驻留wave/CU 6.752001844654996；HBM读取 805629248字节；wait/wavecycle 0.06210353555683649。
- 主循环PC `[{'start_pc': '0x20048', 'end_pc': '0x20384', 'line_start': 220, 'line_end': 357}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[39:40], v[21:22], v[1:4]  // 000000020048: FFD40001 04062B27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[39:40], v[23:24], v[5:8]  // 000000020054: FFD40005 04162F27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[17:18], v[1:4]  // 00000002005C: FFD40001 04062329`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[19:20], v[5:8]  // 000000020064: FFD40005 04162729`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000002006C: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000020070: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000020074: BF8A0000`
  指令PC：`ds_read_b64 v[41:42], v56                                  // 000000020078: D8EC0000 29000038`
  指令PC：`ds_read_m32x16_b16 v[25:28], v57                           // 000000020080: D9F60000 19000039`
  指令PC：`ds_read_b64 v[31:32], v59                                  // 000000020088: D8EC0000 1F00003B`
  指令PC：`ds_read_m32x16_b16 v[17:20], v58                           // 000000020094: D9F60000 1100003A`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s42 idxen offen slc lds// 0000000200AC: E05F3000 2A040021`
  指令PC：`buffer_load_dwordx4 v55, s[12:15], s42 offen lds           // 0000000200BC: E05D1000 2A030037`
  指令PC：`buffer_load_dwordx4 v55, s[12:15], s42 offen lds           // 0000000200CC: E05D1000 2A030037`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 0000000200D4: FFD40001 04061B25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[15:16], v[5:8]  // 0000000200DC: FFD40005 04161F25`
## ID1009 geometry_id1009_pure candidate

- MNKG {'M': 1, 'N': 4096, 'K': 6144, 'G': 16}；实例6034；GPU4/epoch0。
- 文件 `captures/h3low_lds_id1009_candidate_cu0_e9d1ce5e/924385_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsN128Kernel<1>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 18432, 'scratch_bytes': 0, 'vgpr': 60, 'sgpr': 64}。
- 正式candidate/base变化+10.407%；平均驻留wave/CU 10.291163889890612；HBM读取 805834304字节；wait/wavecycle 0.021637123403970906。
- 主循环PC `[{'start_pc': '0x22748', 'end_pc': '0x22a84', 'line_start': 220, 'line_end': 357}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[39:40], v[21:22], v[1:4]  // 000000022748: FFD40001 04062B27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[39:40], v[23:24], v[5:8]  // 000000022754: FFD40005 04162F27`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[41:42], v[17:18], v[1:4]  // 00000002275C: FFD40001 04062329`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[41:42], v[19:20], v[5:8]  // 000000022764: FFD40005 04162729`
  指令PC：`s_waitcnt vmcnt(0)                                         // 00000002276C: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000022770: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000022774: BF8A0000`
  指令PC：`ds_read_b64 v[41:42], v55                                  // 000000022778: D8EC0000 29000037`
  指令PC：`ds_read_m32x16_b16 v[25:28], v56                           // 000000022780: D9F60000 19000038`
  指令PC：`ds_read_b64 v[31:32], v58                                  // 000000022788: D8EC0000 1F00003A`
  指令PC：`ds_read_m32x16_b16 v[17:20], v57                           // 000000022794: D9F60000 11000039`
  指令PC：`buffer_load_dwordx4 v[33:34], s[16:19], s43 idxen offen slc lds// 0000000227AC: E05F3000 2B040021`
  指令PC：`buffer_load_dwordx4 v54, s[12:15], s43 offen lds           // 0000000227BC: E05D1000 2B030036`
  指令PC：`buffer_load_dwordx4 v54, s[12:15], s43 offen lds           // 0000000227CC: E05D1000 2B030036`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[37:38], v[13:14], v[1:4]  // 0000000227D4: FFD40001 04061B25`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[37:38], v[15:16], v[5:8]  // 0000000227DC: FFD40005 04161F25`
