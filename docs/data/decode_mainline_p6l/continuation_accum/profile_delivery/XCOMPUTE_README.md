# N128 四 wave 双 FP32 累加缓冲：当前 CK 与候选

23个预登记三轮配对结束后采集。M17..64固定属性规则和16条实际代表不变；固定620/621/805/335，并补齐存在的收益/平坦/回退类别。两臂来自同一冻结二进制/GPU4/epoch。
候选是未采用实验6035的内部实现修订：同一K32块的两个K16部分分别累加到两套FP32寄存器，末尾逐元素相加一次，再执行原BF16 epilogue。A/B搬运、N128/K32/D2/W4、三套操作数寄存器和selector保持；浮点求和顺序改变，精度阈值不放宽。
保持perf原名及sidecar，Result选择dispatch18、对应kernel、XCD0。target-cu是active-CU索引，GUI的物理CU名称可能不同。
Source查看主循环MMAC的C目标寄存器：两半K应使用分离累加器；再查看收尾FP32相加、DS read、vmcnt/lgkmcnt和barrier。Wavefronts恢复全范围，检查中间25%~75%主循环内MMAC之间依赖/空隙、DS read与MMAC的重叠；尾部查看新增相加。具体绝对Clk在GUI打开后定位。
更多独立累加链能否提高实际发射重叠仍是待验证假设；寄存器增加是明确代价。包内base为当前最佳CK，结构不同，不能称为同二进制纯C-bank开关对照。旧6035跨构建耗时只作诊断，不拼为新成绩。
perf时长不计正式性能成绩，不同wave累计Latency不能相加当kernel墙钟。本机核对trace/dispatch/kernel/代码和完整性，未运行Windows XCompute，GUI尚未确认。

## ID620 id620_accum base

- MNKG {'M': 32, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3accum_id620_base_cu0_ffd60b56/1042569_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+57.594%；平均驻留wave/CU 6.306766853307381；HBM读取 146525632字节；wait/wavecycle 0.15456026574937676。
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
## ID620 id620_accum candidate

- MNKG {'M': 32, 'N': 1024, 'K': 2048, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3accum_id620_candidate_cu0_c3a00f11/1042603_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<2>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 20480, 'scratch_bytes': 0, 'vgpr': 100, 'sgpr': 48}。
- 正式candidate/base变化+57.594%；平均驻留wave/CU 9.59726621677455；HBM读取 140872960字节；wait/wavecycle 0.031377169633293754。
- 主循环PC `[{'start_pc': '0x455f0', 'end_pc': '0x45a48', 'line_start': 255, 'line_end': 427}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[73:74], v[45:46], v[17:20]  // 0000000455F0: FFD40011 04465B49`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[73:74], v[47:48], v[21:24]  // 0000000455F8: FFD40015 04565F49`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[71:72], v[45:46], v[5:8]  // 000000045600: FFD40005 04165B47`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[71:72], v[47:48], v[1:4]  // 000000045608: FFD40001 04065F47`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[69:70], v[41:42], v[29:32]  // 000000045610: FFD4001D 04765345`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[69:70], v[43:44], v[25:28]  // 000000045618: FFD40019 04665745`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[67:68], v[41:42], v[13:16]  // 000000045620: FFD4000D 04365343`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[67:68], v[43:44], v[9:12]  // 000000045628: FFD40009 04265743`
  指令PC：`s_waitcnt vmcnt(0)                                         // 000000045630: BF8C0F70`
  指令PC：`s_waitcnt lgkmcnt(0)                                       // 000000045634: BF8CC07F`
  指令PC：`s_barrier                                                  // 000000045638: BF8A0000`
  指令PC：`ds_read_b64 v[75:76], v93                                  // 00000004563C: D8EC0000 4B00005D`
  指令PC：`ds_read_b64 v[71:72], v94                                  // 000000045644: D8EC0000 4700005E`
  指令PC：`ds_read_m32x16_b16 v[49:52], v95                           // 00000004564C: D9F60000 3100005F`
  指令PC：`ds_read_b64 v[67:68], v98                                  // 000000045654: D8EC0000 43000062`
  指令PC：`ds_read_b64 v[55:56], v97                                  // 00000004565C: D8EC0000 37000061`
## ID621 id621_accum base

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例5003；GPU4/epoch0。
- 文件 `captures/h3accum_id621_base_cu0_49a523ed/1042643_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化-12.069%；平均驻留wave/CU 6.370573217096212；HBM读取 155776256字节；wait/wavecycle 0.14296614318283304。
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
## ID621 id621_accum candidate

- MNKG {'M': 64, 'N': 1024, 'K': 2048, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3accum_id621_candidate_cu0_4366e2bd/1042677_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 164, 'sgpr': 64}。
- 正式candidate/base变化-12.069%；平均驻留wave/CU 6.212262717486678；HBM读取 142906368字节；wait/wavecycle 0.0918119706424847。
- 主循环PC `[{'start_pc': '0x6fdf4', 'end_pc': '0x7048c', 'line_start': 309, 'line_end': 553}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[49:52], v[121:122], v[77:78], v[49:52]  // 00000006FDF4: FFD40031 04C69B79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[53:56], v[121:122], v[79:80], v[53:56]  // 00000006FDFC: FFD40035 04D69F79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[37:40], v[119:120], v[77:78], v[37:40]  // 00000006FE04: FFD40025 04969B77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[33:36], v[119:120], v[79:80], v[33:36]  // 00000006FE0C: FFD40021 04869F77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[117:118], v[77:78], v[21:24]  // 00000006FE14: FFD40015 04569B75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[117:118], v[79:80], v[17:20]  // 00000006FE1C: FFD40011 04469F75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[115:116], v[77:78], v[5:8]  // 00000006FE24: FFD40005 04169B73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[115:116], v[79:80], v[1:4]  // 00000006FE2C: FFD40001 04069F73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[61:64], v[113:114], v[73:74], v[61:64]  // 00000006FE34: FFD4003D 04F69371`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[57:60], v[113:114], v[75:76], v[57:60]  // 00000006FE3C: FFD40039 04E69771`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[45:48], v[109:110], v[73:74], v[45:48]  // 00000006FE44: FFD4002D 04B6936D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[41:44], v[109:110], v[75:76], v[41:44]  // 00000006FE4C: FFD40029 04A6976D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[107:108], v[73:74], v[29:32]  // 00000006FE54: FFD4001D 0476936B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[107:108], v[75:76], v[25:28]  // 00000006FE5C: FFD40019 0466976B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[105:106], v[73:74], v[13:16]  // 00000006FE64: FFD4000D 04369369`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[105:106], v[75:76], v[9:12]  // 00000006FE6C: FFD40009 04269769`
## ID805 id805_accum base

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例5000；GPU4/epoch0。
- 文件 `captures/h3accum_id805_base_cu0_f7f80836/1042721_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7Uniform<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<256, 256, 64>, ck_tile::sequence<4, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, (ck_tile::V7UniformKind)0>, ck_tile::GroupedGemmV7UniformStoreEpilogue<unsigned short, ck_tile::V7UniformAccTile, ck_tile::V7UniformGeometry>, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 256, 'lds_bytes': 65536, 'scratch_bytes': 0, 'vgpr': 212, 'sgpr': 48}。
- 正式candidate/base变化+13.668%；平均驻留wave/CU 6.577135394755756；HBM读取 559132544字节；wait/wavecycle 0.170513194263459。
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
## ID805 id805_accum candidate

- MNKG {'M': 64, 'N': 2048, 'K': 4096, 'G': 32}；实例6035；GPU4/epoch0。
- 文件 `captures/h3accum_id805_candidate_cu0_42fe9f64/1042775_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [131072, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 512, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 164, 'sgpr': 64}。
- 正式candidate/base变化+13.668%；平均驻留wave/CU 6.854758606991665；HBM读取 566486592字节；wait/wavecycle 0.1261939179834916。
- 主循环PC `[{'start_pc': '0x6fdf4', 'end_pc': '0x7048c', 'line_start': 309, 'line_end': 553}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[49:52], v[121:122], v[77:78], v[49:52]  // 00000006FDF4: FFD40031 04C69B79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[53:56], v[121:122], v[79:80], v[53:56]  // 00000006FDFC: FFD40035 04D69F79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[37:40], v[119:120], v[77:78], v[37:40]  // 00000006FE04: FFD40025 04969B77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[33:36], v[119:120], v[79:80], v[33:36]  // 00000006FE0C: FFD40021 04869F77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[117:118], v[77:78], v[21:24]  // 00000006FE14: FFD40015 04569B75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[117:118], v[79:80], v[17:20]  // 00000006FE1C: FFD40011 04469F75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[115:116], v[77:78], v[5:8]  // 00000006FE24: FFD40005 04169B73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[115:116], v[79:80], v[1:4]  // 00000006FE2C: FFD40001 04069F73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[61:64], v[113:114], v[73:74], v[61:64]  // 00000006FE34: FFD4003D 04F69371`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[57:60], v[113:114], v[75:76], v[57:60]  // 00000006FE3C: FFD40039 04E69771`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[45:48], v[109:110], v[73:74], v[45:48]  // 00000006FE44: FFD4002D 04B6936D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[41:44], v[109:110], v[75:76], v[41:44]  // 00000006FE4C: FFD40029 04A6976D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[107:108], v[73:74], v[29:32]  // 00000006FE54: FFD4001D 0476936B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[107:108], v[75:76], v[25:28]  // 00000006FE5C: FFD40019 0466976B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[105:106], v[73:74], v[13:16]  // 00000006FE64: FFD4000D 04369369`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[105:106], v[75:76], v[9:12]  // 00000006FE6C: FFD40009 04269769`
## ID335 id335_accum base

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例5003；GPU4/epoch0。
- 文件 `captures/h3accum_id335_base_cu0_6000f27e/1042829_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 2, ck_tile::GroupedGemmV7UniformKernel<ck_tile::GroupedGemmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, 1, 1>, ck_tile::GemmPipelineV7SmallM<ck_tile::UniversalGemmPipelineProblem<unsigned short, unsigned short, float, ck_tile::TileGemmShape<ck_tile::sequence<64, 128, 64>, ck_tile::sequence<2, 2, 1>, ck_tile::sequence<16, 16, 16>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, false, false, true, true>, (ck_tile::GemmPipelineScheduler)1, true, (ck_tile::TailNumber)10, ck_tile::null_type, ck_tile::null_type, false, 2, 2>, 128>, ck_tile::GroupedGemmV7SmallMStoreEpilogue<unsigned short, ck_tile::V7SmallMAccTile<ck_tile::V7SmallMGeometry<128> >, ck_tile::V7SmallMGeometry<128> >, (ck_tile::GroupedGemmMFilter)0, 0> >, void const AS4*, int, int>(void const AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 32768, 'scratch_bytes': 0, 'vgpr': 80, 'sgpr': 48}。
- 正式candidate/base变化+0.326%；平均驻留wave/CU 5.995627657607691；HBM读取 139344768字节；wait/wavecycle 0.1407910190125284。
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
## ID335 id335_accum candidate

- MNKG {'M': 64, 'N': 4096, 'K': 2048, 'G': 8}；实例6035；GPU4/epoch0。
- 文件 `captures/h3accum_id335_candidate_cu0_82ccedde/1042863_tile_example_grouped_gemm_device_args_benchmark.perf`；dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<256, 1, ck_tile::DecodeNnLowLdsWave4N128Kernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [65536, 1, 1], 'block_threads': [256, 1, 1], 'ctas': 256, 'lds_bytes': 24576, 'scratch_bytes': 0, 'vgpr': 164, 'sgpr': 64}。
- 正式candidate/base变化+0.326%；平均驻留wave/CU 6.106713216480547；HBM读取 136325888字节；wait/wavecycle 0.12839543657798788。
- 主循环PC `[{'start_pc': '0x6fdf4', 'end_pc': '0x7048c', 'line_start': 309, 'line_end': 553}]`。

  指令PC：`v_mmac_f32_16x16x16_bf16 v[49:52], v[121:122], v[77:78], v[49:52]  // 00000006FDF4: FFD40031 04C69B79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[53:56], v[121:122], v[79:80], v[53:56]  // 00000006FDFC: FFD40035 04D69F79`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[37:40], v[119:120], v[77:78], v[37:40]  // 00000006FE04: FFD40025 04969B77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[33:36], v[119:120], v[79:80], v[33:36]  // 00000006FE0C: FFD40021 04869F77`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[21:24], v[117:118], v[77:78], v[21:24]  // 00000006FE14: FFD40015 04569B75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[17:20], v[117:118], v[79:80], v[17:20]  // 00000006FE1C: FFD40011 04469F75`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[5:8], v[115:116], v[77:78], v[5:8]  // 00000006FE24: FFD40005 04169B73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[1:4], v[115:116], v[79:80], v[1:4]  // 00000006FE2C: FFD40001 04069F73`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[61:64], v[113:114], v[73:74], v[61:64]  // 00000006FE34: FFD4003D 04F69371`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[57:60], v[113:114], v[75:76], v[57:60]  // 00000006FE3C: FFD40039 04E69771`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[45:48], v[109:110], v[73:74], v[45:48]  // 00000006FE44: FFD4002D 04B6936D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[41:44], v[109:110], v[75:76], v[41:44]  // 00000006FE4C: FFD40029 04A6976D`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[29:32], v[107:108], v[73:74], v[29:32]  // 00000006FE54: FFD4001D 0476936B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[25:28], v[107:108], v[75:76], v[25:28]  // 00000006FE5C: FFD40019 0466976B`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[13:16], v[105:106], v[73:74], v[13:16]  // 00000006FE64: FFD4000D 04369369`
  指令PC：`v_mmac_f32_16x16x16_bf16 v[9:12], v[105:106], v[75:76], v[9:12]  // 00000006FE6C: FFD40009 04269769`
