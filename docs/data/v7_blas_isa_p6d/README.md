# P6d BLAS ISA-guided diagnosis

结论与下一阶段候选见 [report.md](report.md)。九个代表shape，包含新增M128/M256及大M大K；不是1193全量。本阶段没有修改CK实现和默认selector。

`isa_features.json`只保存自己的解析结果、资源元数据、符号/哈希和控制流计数，不包含BLAS机器码或反汇编正文；raw `.s`、code object、`.perf`/SQTT在`/workspace/v7_coverage/p6d/`。`coverage.json`说明选定六个BLAS内核在历史1193条映射中的覆盖，不表示对724条做了GPU复测。`timing.json`保留每轮中位数、CV和全局范围；Native repeat1/20分别列出。`profiles.json`是本次13组计数，`historical_profiles.json`为明确标注的P6a/P6b证据复用。`raw_counters.json`保留每个有效profile的两个dispatch。

`jobs.jsonl.gz`/`gate.jsonl.gz`/`logs.json.gz`为完整命令、门禁、结果与日志的无损压缩。现场依赖由manifest和script_tool_sha256固定；脚本是现场环境审计副本，重放需原路径的工具/BLAS/benchmark。没有复制BLAS实现进CK；没有push。
