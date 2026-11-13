# 八波K32/K64直接控制SQTT

固定ID805，M64/N2048/K4096/G32，GPU4，同binary/epoch。base是实验K32D4八波6027，candidate是K64D2八波6029；两者均为完整M所有权/N128，不把base称为当前最佳CK。
先完整解压保留原文件名及sidecar，分别打开下面perf，Result选择dispatch18、对应kernel、XCD0。GUI物理CU标签可能与active-CU索引不同。
Summary比较实际VGPR/LDS；Source定位重复K循环的buffer_load、vmcnt、ds_read、lgkmcnt、barrier和MMAC；Wavefronts恢复完整范围，重点比较25%～75%稳态的空档。
K64相对K32的三轮发现点估计约慢15%，不是七block确认。K和全局stage数/资源一同改变，不能把差异只归因一条wait。平均驻留和累计wait不是单独因果证明，也不能把跨wave累计时钟相加当kernel墙钟。
这些perf/SQTT耗时仅用于诊断，不计正式性能成绩。本机仅检查文件、实际代码、dispatch和trace，未打开Windows XCompute。

## base 实例6027

- 文件 `captures/h3k64_id805_base_cu0_8045aa8e/697523_tile_example_grouped_gemm_device_args_benchmark.perf`，dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeWidePhaseKernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [262144, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 512, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 112, 'sgpr': 64}；平均驻留wave/CU 7.382857627771173，wait/wavecycle 0.16990326265009237，HBM读取 557622400字节。
- 主循环PC `[{'start_pc': '0x4b488', 'end_pc': '0x4bdc0', 'static_mmac': 96, 'line_start': 283, 'line_end': 671}]`。

## candidate 实例6029

- 文件 `captures/h3k64_id805_candidate_cu0_b97a425b/697577_tile_example_grouped_gemm_device_args_benchmark.perf`，dispatch18/XCD0/active-CU-index0。
- kernel `void ck_tile::kentry<512, 1, ck_tile::DecodeNnCooperativeK64WideKernel<4>, void AS4*, int, int>(void AS4*, int, int)`。
- 资源 {'grid_work_items': [262144, 1, 1], 'block_threads': [512, 1, 1], 'ctas': 512, 'lds_bytes': 49152, 'scratch_bytes': 0, 'vgpr': 168, 'sgpr': 64}；平均驻留wave/CU 7.210188306425802，wait/wavecycle 0.25551973197410555，HBM读取 562838848字节。
- 主循环PC `[{'start_pc': '0x50980', 'end_pc': '0x5107c', 'line_start': 275, 'line_end': 531}]`。
