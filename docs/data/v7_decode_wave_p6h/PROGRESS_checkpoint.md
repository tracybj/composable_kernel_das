

### P6h 独立decode架构原型阶段（2026-11-06；实验用，不合入）

本轮从用户“重新写一条有效合理的流水”开始，为独立普通任务，不恢复已暂停的P6g Goal、不重置旧复现期限。T0为2026-11-06T05:57:18UTC。用户追问A放LDS/B放寄存器后补做该对照；截至本阶段，尚未形成值得整域替代现有最优方案的新流水，不能宣称目标已达成。依CLAUDE.md阶段规则完成本地原型记录后停止，不进入下一轮重写。

- 新实例6000、独立decode family/pipeline/provider/manifest/CMake archive，默认关闭，不依赖V7 enable，不实例化V7 pipeline/epilogue。最初四个wave私有B-LDS/M16双缓冲；后隔离A cache-swizzle、8级预取、单wave全N128/M16/32/64，最后补A-LDS/B-register K32两级。各版本与负结果均保留，不跨二进制取赢家；canonical suffix仍为首版遗留别名，后续正式集成前须泛化名称并重建复验，当前不是产品推广。
- 每版预声明22条代表（含所有新增1179–1193），14活动/8回退。初版和cache-swizzle版14条全回退；8级版8改善/6回退，中位−3.92%，但不是可推广属性域；单wave版1改善/9回退/4在±2%，中位+3.42%。A-LDS最终14活动全回退，中位+18.70%，12条三轮均回退>2%、9条至少一轮CV>5%。100条预声明扩围未执行，因代表域表现不足，不按赢家ID扩围。
- ID772初版B读重复HBM约9.2GB，对照2.5GB；全M所有权降回2.5GB但没有自动提速。A-LDS最终ID633 VMEM/wave207.5→1089、LDS/wave420→137，HBM34.23→33.69MB接近，仍+66.81%；ID772 VMEM/wave462→2244、HBM2527.29→2525.72MB。当前B标量加载/打包、保守等待和A LDS冲突存在代价，不能据此排除更好的B合并加载/寄存器重排或共享A调度。
- 最后源码A-LDS/B-register：四个N-wave，每wave完整M16/32/64×32输出，K32两级，A私有LDS/B直读VGPR，0 CTA barrier/无跨wave归约，gfx936 VGPR95/118/107、SGPR51/53/57，LDS8/16/32KiB，scratch/spill0。只验收正式benchmark packed输入；任意stride等完整生产契约未验收，不推广默认启用。v1布局/语义未改，仅追加ID/family。
- 最后官方binary SHA276fdae07237401fdfb56976c7c6c11faa61418deaf8097e4839607da1fbfc99；19合成边界/压力（含M31/33/49、最小K、奇数K32阶段、G1024、大N/最大K）及22正式shape，63精度作业/410次全输出/3,417,443,328元素max_abs0、CK/BLAS各32guard通过；双方精度先于三轮相邻A/B。selector/query/smoke、7字节边界、CPU新选核64800/旧策略等价241920/两表2386可达性通过，原18个V7 shader字节不变。
- 997 raw作业仅GPU4串行、epoch0，无MKFD/服务事件、无恢复/重置/换卡/终止其他进程。4失败保留：query错误继承V7开关、profile过滤未捕获、MMAC提前到LDS等待前、opaque-asm VMEM结果等待前复制；后两项精度失败已修复并换SHA完整补测，不伪装环境失败。编译/资源spill拒绝和CPU permlane16探针不支持均保留。
- 正式benchmark复用暂停P6g已准备的默认关闭真实redzone/非有限值拒绝功能，修复float-equal构建错误；计时显式关闭redzone，未恢复P6g。两份shape SHA仍5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863，用户readfirstlane diff MD5仍4c7f49a5c7592767d53195a713db6b8a、未暂存。暂停任务tools/grouped_gemm_reproduce保留不纳入提交。
- 报告/workspace/v7_coverage/p6h_decode_arch/report.md；Git归档docs/data/v7_decode_wave_p6h/，含源码版本、全结果/负结果、所有命令/环境/SHA/健康门禁及无损压缩raw日志。binary/vendor ISA/巨型perf留现场。下一阶段候选是B合并加载与寄存器重排、共享A调度，尚未实施；现有最优方案保持不变。
