# N128 四 wave A缓存标志修订：当前 CK 与候选

23个预登记三轮配对完整结束后采集。单一M17..64属性规则，全部16条实际代表；固定620/621/805/335，按几何选择可trace的大CTA样本，并补齐实际存在的收益/平坦/回退类别。低CTA形状仍在全部精度/计时/最终域，不因trace困难排除。每组base均为当前最佳CK，candidate为6035的A缓存标志修订；旧6035历史耗时不拼入当前成绩。当前新二进制内两臂是正式对照，跨构建cache旧/新仅作静态ISA及诊断比较。

完整解压并保持原名及sidecar。XCompute打开下列perf，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI物理CU标签可能不同。
先在Summary核对block/grid/VGPR/LDS，再查看Source重复K主循环的A/B Direct-to-LDS、ds_read、vmcnt/lgkmcnt、barrier和MMAC。Wavefronts恢复全范围，依次看0～15%、25～75%、最后15%的相对窗口；准确绝对Clk需打开后确定。
候选6035相对其先前修订仅改变A-load显式SLC修饰，静态探针指令与资源证明另存。当前包内base为P6k当前最佳CK，分块/指令与候选不同，不能把它称为同二进制纯缓存位对照。比较全dispatch与每CTA的实际计数、HBM、等待和时间，核对候选A-load不带SLC。旧6035耗时来自不同构建，只作诊断，不拼成新成绩；不同wave累计Latency不相加当kernel墙钟。
本包的candidate/base为三轮代表发现值，尚不是七block采用成绩；perf/SQTT时长仅作诊断。本机核对文件/dispatch/实际代码/trace字节，未运行Windows XCompute，不宣称GUI已验证。

## ID620 id620_noslc base

- MNKG {'M': 32, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3noslc_id620_base_cu0_8018c6f2/1010768_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+59.558%；平均驻留wave/CU 6.1614699613374535；HBM读取 145968512字节；wait/wavecycle 0.15218972479769816。
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
## ID620 id620_noslc candidate

- MNKG {'M': 32, 'N': 1024, 'K': 2048, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3noslc_id620_candidate_cu0_8116be91/1010808_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 20480, 'scratch_bytes': 0, 'vgpr': 84, 'sgpr': 48}。
- 正式candidate/base变化+59.558%；平均驻留wave/CU 9.649804040794097；HBM读取 141004416字节；wait/wavecycle 0.03607734869173125。
- 主循环PC `[{'start_pc': '0x455b0', 'end_pc': '0x45a08', 'line_start': 239, 'line_end': 411}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[57:58], v[29:30], v[9:12]  // 0000000455B0: FFD40009 04263B39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[57:58], v[31:32], v[13:16]  // 0000000455B8: FFD4000D 04363F39`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[55:56], v[29:30], v[5:8]  // 0000000455C0: FFD40005 04163B37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[55:56], v[31:32], v[1:4]  // 0000000455C8: FFD40001 04063F37`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[53:54], v[25:26], v[9:12]  // 0000000455D0: FFD40009 04263335`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[53:54], v[27:28], v[13:16]  // 0000000455D8: FFD4000D 04363735`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[51:52], v[25:26], v[5:8]  // 0000000455E0: FFD40005 04163333`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[51:52], v[27:28], v[1:4]  // 0000000455E8: FFD40001 04063733`
  指令PC：`s_waitcnt vmcnt(0)                                         // 0000000455F0: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 0000000455F4: BF8CC07F`
  指令PC：`s_barrier                                                  // 0000000455F8: BF8A0000`
  指令PC：`ds_read_b64 v[59:60], v77                                  // 0000000455FC: D8EC0000 3B00004D`
  指令PC：`ds_read_b64 v[55:56], v78                                  // 000000045604: D8EC0000 3700004E`
  指令PC：`ds_read_m32x16_b16 v[33:36], v79                           // 00000004560C: D9F60000 2100004F`
  指令PC：`ds_read_b64 v[51:52], v82                                  // 000000045614: D8EC0000 33000052`
  指令PC：`ds_read_b64 v[39:40], v81                                  // 00000004561C: D8EC0000 27000051`
## ID621 id621_noslc base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3noslc_id621_base_cu0_493abc35/1010842_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-12.690%；平均驻留wave/CU 6.366944753219452；HBM读取 156363456字节；wait/wavecycle 0.16471016005375297。
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
## ID621 id621_noslc candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3noslc_id621_candidate_cu0_44af6823/1010882_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 132, 'sgpr': 64}。
- 正式candidate/base变化-12.690%；平均驻留wave/CU 6.182356292990002；HBM读取 142683520字节；wait/wavecycle 0.09473696323430697。
- 主循环PC `[{'start_pc': '0x6fc74', 'end_pc': '0x7030c', 'line_start': 277, 'line_end': 521}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 00000006FC74: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 00000006FC7C: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[87:88], v[45:46], v[21:24]  // 00000006FC84: FFD40015 04565B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[87:88], v[47:48], v[17:20]  // 00000006FC8C: FFD40011 04465F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[85:86], v[45:46], v[13:16]  // 00000006FC94: FFD4000D 04365B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[85:86], v[47:48], v[9:12]  // 00000006FC9C: FFD40009 04265F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[45:46], v[5:8]  // 00000006FCA4: FFD40005 04165B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[47:48], v[1:4]  // 00000006FCAC: FFD40001 04065F53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[81:82], v[41:42], v[25:28]  // 00000006FCB4: FFD40019 04665351`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[81:82], v[43:44], v[29:32]  // 00000006FCBC: FFD4001D 04765751`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[79:80], v[41:42], v[21:24]  // 00000006FCC4: FFD40015 0456534F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[79:80], v[43:44], v[17:20]  // 00000006FCCC: FFD40011 0446574F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[77:78], v[41:42], v[13:16]  // 00000006FCD4: FFD4000D 0436534D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[77:78], v[43:44], v[9:12]  // 00000006FCDC: FFD40009 0426574D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[75:76], v[41:42], v[5:8]  // 00000006FCE4: FFD40005 0416534B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[75:76], v[43:44], v[1:4]  // 00000006FCEC: FFD40001 0406574B`
## ID805 id805_noslc base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3noslc_id805_base_cu0_a489526b/1010916_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+13.118%；平均驻留wave/CU 6.670304840058118；HBM读取 567540096字节；wait/wavecycle 0.1717610255570635。
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
## ID805 id805_noslc candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3noslc_id805_candidate_cu0_c1c819f7/1010984_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 132, 'sgpr': 64}。
- 正式candidate/base变化+13.118%；平均驻留wave/CU 6.708081603903354；HBM读取 565660352字节；wait/wavecycle 0.09460607565832625。
- 主循环PC `[{'start_pc': '0x6fc74', 'end_pc': '0x7030c', 'line_start': 277, 'line_end': 521}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 00000006FC74: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 00000006FC7C: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[87:88], v[45:46], v[21:24]  // 00000006FC84: FFD40015 04565B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[87:88], v[47:48], v[17:20]  // 00000006FC8C: FFD40011 04465F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[85:86], v[45:46], v[13:16]  // 00000006FC94: FFD4000D 04365B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[85:86], v[47:48], v[9:12]  // 00000006FC9C: FFD40009 04265F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[45:46], v[5:8]  // 00000006FCA4: FFD40005 04165B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[47:48], v[1:4]  // 00000006FCAC: FFD40001 04065F53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[81:82], v[41:42], v[25:28]  // 00000006FCB4: FFD40019 04665351`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[81:82], v[43:44], v[29:32]  // 00000006FCBC: FFD4001D 04765751`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[79:80], v[41:42], v[21:24]  // 00000006FCC4: FFD40015 0456534F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[79:80], v[43:44], v[17:20]  // 00000006FCCC: FFD40011 0446574F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[77:78], v[41:42], v[13:16]  // 00000006FCD4: FFD4000D 0436534D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[77:78], v[43:44], v[9:12]  // 00000006FCDC: FFD40009 0426574D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[75:76], v[41:42], v[5:8]  // 00000006FCE4: FFD40005 0416534B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[75:76], v[43:44], v[1:4]  // 00000006FCEC: FFD40001 0406574B`
## ID335 id335_noslc base

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例5003；GPU4/epoch0。
- 文件 `captures/h3noslc_id335_base_cu0_5c158161/1011047_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+1.120%；平均驻留wave/CU 5.950051035630722；HBM读取 139310016字节；wait/wavecycle 0.15183812663172447。
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
## ID335 id335_noslc candidate

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例6035；GPU4/epoch0。
- 文件 `captures/h3noslc_id335_candidate_cu0_32a55b89/1011081_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 132, 'sgpr': 64}。
- 正式candidate/base变化+1.120%；平均驻留wave/CU 6.017092058416053；HBM读取 136318208字节；wait/wavecycle 0.12312126852848791。
- 主循环PC `[{'start_pc': '0x6fc74', 'end_pc': '0x7030c', 'line_start': 277, 'line_end': 521}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[89:90], v[45:46], v[25:28]  // 00000006FC74: FFD40019 04665B59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[89:90], v[47:48], v[29:32]  // 00000006FC7C: FFD4001D 04765F59`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[87:88], v[45:46], v[21:24]  // 00000006FC84: FFD40015 04565B57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[87:88], v[47:48], v[17:20]  // 00000006FC8C: FFD40011 04465F57`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[85:86], v[45:46], v[13:16]  // 00000006FC94: FFD4000D 04365B55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[85:86], v[47:48], v[9:12]  // 00000006FC9C: FFD40009 04265F55`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[83:84], v[45:46], v[5:8]  // 00000006FCA4: FFD40005 04165B53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[83:84], v[47:48], v[1:4]  // 00000006FCAC: FFD40001 04065F53`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[81:82], v[41:42], v[25:28]  // 00000006FCB4: FFD40019 04665351`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[81:82], v[43:44], v[29:32]  // 00000006FCBC: FFD4001D 04765751`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[79:80], v[41:42], v[21:24]  // 00000006FCC4: FFD40015 0456534F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[79:80], v[43:44], v[17:20]  // 00000006FCCC: FFD40011 0446574F`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[77:78], v[41:42], v[13:16]  // 00000006FCD4: FFD4000D 0436534D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[77:78], v[43:44], v[9:12]  // 00000006FCDC: FFD40009 0426574D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[75:76], v[41:42], v[5:8]  // 00000006FCE4: FFD40005 0416534B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[75:76], v[43:44], v[1:4]  // 00000006FCEC: FFD40001 0406574B`
