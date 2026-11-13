# Decode 优先策略：减少无效回退检查

该修复属于既有 host-policy 固定开销因素：当几何不满足任何请求的新 decode 契约时，跳过一次必然失败的优先 selector 调用，再执行原完整策略。没有新增流水、shape 特判、环境缓存、公开 ABI 或选核结果变化。

benchmark SHA256：`e5105ceaf0c45ca6df3b7bcab390a97f29a906c147fa050f44c7a27dcfba8472`。1766个GPU shader的完整机器码和资源记录与修复前相同。322845项CPU对照的status/instance/reason全部相同，包含全部64种public-disable组合。

35项资格、10组三轮配对、159次全输出和真实guard全部通过。固定清单为7个原回退控制与620/621/805三条主线对照；全部配置、实际实例、轮次、TFLOPS、R_frozen与高CV保留。

三条主线仍使用已被拒绝的6035候选，只用于检查host修复没有改变流水，不构成整域性能采用。没有抽取新的盲测seed，也没有把这10组或历史跨构建赢家拼入正式表。50/30计数仍为0/0。

CPU抽取策略体的计时仅是辅助诊断，排除了架构检测、HIP启动和GPU；其绝对时间不能直接等同GPU事件耗时。旧/新GPU二进制的差值不冒充同二进制纯host修复收益。

本阶段无需重复捕获机器码完全相同的SQTT；先前第15批仍为其自身二进制的诊断证据。正式benchmark包含默认selector/run适配器，include_args_build0没有移除host策略解析，短kernel时间需考虑这部分影响。

P6k、冻结BLAS、1193shape和原Excel不变；只做本地实验提交，不push、不合入。编译输入26份保留现场，未将用户overlay源文件代为提交。
