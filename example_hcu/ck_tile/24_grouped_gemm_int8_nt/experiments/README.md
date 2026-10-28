# 历史候选



`packed_v12` 保存整合前的候选模板和全部实验 ID，包含未获收益的 LDS 双缓冲、宽 LDS、padding 与 wave 划分对照。它们不进入公开 selector，也不参与默认 CMake 构建。实际编译产物、命令、资源和数值/性能数据保留在服务器本 example 的 build/iteration_* 和 results/probe_*。



重放此快照时，将同目录的 instances.cpp 和 grouped_int8_nt_tile.hpp 与本 example 的 direct.hpp、probe_bridge.cpp 配套构建；头文件中的模板不依赖当前精简的 TileProblem。请使用该次记录中的工具链、编译参数和 shape；历史编号不属于公开 API。
