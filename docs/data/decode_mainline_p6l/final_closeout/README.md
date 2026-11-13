# P6l 最终审计归档：目标未达成

本轮没有采用的新方案；稳定改善0/50、可归因跨桶0/30。完整报告见[closeout/README.md](closeout/README.md)，逐项状态见[closeout/requirements_audit.json](closeout/requirements_audit.json)。

408个三轮发现配对与166个七block原表确认均保留。后者GM candidate/base=1.053153，105条中位回退>2%，94条低CV且全部七block回退>2%，整域拒绝。28条个体稳定/14条个体跨桶不构成采用，不回填正式表。短线发现有负结果，最终确认未完成。最后提前计算MMAC原型只有编译/CPU/静态证据，见[phase_compiled_only](../phase_compiled_only/README.md)。

6632个已启动GPU作业全部终态，原始输出、门禁和版本逐项核对；全输出检查失败元素0，不代表性能验收通过。八批perf包见evidence_index.json及报告下载链接，保留Source/Wavefronts定位；本机未打开Windows XCompute。原P6k、冻结BLAS、1193 shape和旧Excel不变，projection文件明确是历史基线未替换，不是1193新鲜全量。

原新实验截止2026-11-12 22:53:23 UTC、硬截止2026-11-13 02:53:23 UTC保持。构建完成后的GPU接续没有启动，原型未验证；未归咎于等待用户或MKFD。恢复过一次是dmesg权限问题，按标准两次健康观察与固定探针处理，没有已确认的MKFD硬件事件。

evidence_index.json列出此前本地实验提交与8份perf包；prior_archive_Git_blobs.json证明原归档文件与当前Git对象一致。大型perf、二进制、厂商ISA及实际编译输入中的用户overlay留在现场，未新增到本归档。当前9处用户tracked及10份untracked内容保持，readfirstlane受保护，未push。原执行入口/控制器没有获得新时窗；未来实验必须另行授权，不能重置旧session。Goal仍未完成。
