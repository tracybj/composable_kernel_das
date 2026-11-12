# 第一批SQTT：优先查看的K主循环

下列PC来自实际捕获代码的静态后向分支。请在Source按Hit Count确认该分支确实执行，并在Wavefronts放大中段重复K循环。泛型CK还有包住参数解码与写回的外层tile循环，此处优先列内层K循环，避免停在最开头的s_load等待。

|ID|方案|代码对象PC区间|相对kernel起点|
|---:|---|---|---|
|76|current_best|0x2ae0 ～ 0x37f8|+0x9e0 ～ +0x16f8|
|76|frozen_native|0xeccd860 ～ 0xeccdb94|+0xa60 ～ +0xd94|
|621|current_best|0x2a70 ～ 0x3310|+0x970 ～ +0x1210|
|621|frozen_native|0x25d3798 ～ 0x25d3bfc|+0xb98 ～ +0xffc|
|805|current_best|0x2c6c ～ 0x3f3c|+0xb6c ～ +0x1e3c|
|805|frozen_native|0xec5c4c0 ～ 0xec5c974|+0xbc0 ～ +0x1074|
|1072|current_best|0x2d20 ～ 0x3ff0|+0xc20 ～ +0x1ef0|
|1072|frozen_native|0xec53d34 ～ 0xec541e8|+0xb34 ～ +0xfe8|

先查这些区间内的buffer_load、ds_read、s_waitcnt、s_barrier、v_mmac；比较当前MMAC使用的寄存器与提前ds_read的目标寄存器是否不同。长条和累计Latency不直接证明某类操作决定总耗时。

此文件是原下载包之外的补充定位，原包及其SHA保持不变。本机未运行Windows XCompute，GUI中的具体时间区间仍需实际查看。