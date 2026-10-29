# Iteration96：全部344条大M的cached INT8权重与raw线性寻址对照（运行中）

用户允许专家权重加载时一次预排并复用，要求FP16/BF16接口不受影响。此轮仅INT8×INT8→INT32/no-scale/NT，保持原始344条M>=256的M/N/K/G；物理卡7。六个候选：同构建公开v80默认实例、四个完整raw线性B映射/record策略、原方向静态B预排。预排不涉及动态激活A，准备GPU时间和额外字节另列。现有公共源码保持v80原样，没有改FP16/BF16参数/布局/默认路由。

冻结v90/v92已通过96+300项全范围前置检查的库；每条每候选三个HIP-event样本、warmup2/repeat5，并按shape/sample轮换次序。使用精确FP32参考（输入[-8,8]、TF32关闭），逐候选完整输出比较、guard及输入采样不变性检查。总计344×6=2064条开发记录；不是新的1178条公开API验收。numeric/progress/done/error状态见results/packed_linear_screen_v96。

后续新增显式INT8 cached-weight入口，与旧FP16/BF16接口、结构、权重布局及默认路径隔离；生产分流按M/N/K/G，不按caseID。未获得全域证据前不修改正式975/1178计数。
